from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchtext.data import Field, Dataset, Example, BucketIterator
import gensim

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from datasets import load_dataset, load_metric

import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, classification_report
from seqeval.metrics import f1_score as seqeval_f1_score

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import seaborn as sns

from transformers import AutoModelForTokenClassification
import transformers

from colorama import Back, Style

# ignore every warning
import warnings

warnings.filterwarnings("ignore")


TEXT2ID = {
    "B-O": 0,
    "B-AC": 1,
    "PAD": 2,
    "B-LF": 3,
    "I-LF": 4,
}
ID2TEXT = {v: k for k, v in TEXT2ID.items()}


def preprocess(df):
    df = df.drop(columns=["pos_tags"])
    df = df.rename(columns={"ner_tags": "labels"})
    df["ids"] = df["labels"].apply(lambda x: [TEXT2ID[i] for i in x])
    df["sentences"] = df["tokens"].apply(lambda x: " ".join(x))

    return df


def init_data(train_dataset, val_dataset, test_dataset, batch_size=16):

    text_field = Field(
        sequential=True, tokenize=lambda x: x, include_lengths=True
    )  # Default behaviour is to tokenize by splitting
    label_field = Field(sequential=True, tokenize=lambda x: x, is_target=True)

    fields = {"sentences": ("text", text_field), "ids": ("label", label_field)}

    def read_data(df):
        examples = []
        fields = {
            "sentence_labels": ("labels", label_field),
            "sentence_tokens": ("text", text_field),
        }

        for i in range(len(df)):
            tokens = df["tokens"][i]
            labels = df["labels"][i]

            e = Example.fromdict(
                {"sentence_labels": labels, "sentence_tokens": tokens}, fields=fields
            )
            examples.append(e)

        return Dataset(examples, fields=[("labels", label_field), ("text", text_field)])

    train_data = read_data(train_dataset)
    val_data = read_data(val_dataset)
    test_data = read_data(test_dataset)

    VOCAB_SIZE = 20000

    text_field.build_vocab(train_data, max_size=VOCAB_SIZE)
    label_field.build_vocab(train_data)

    BATCH_SIZE = 16
    train_iter = BucketIterator(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
    )
    val_iter = BucketIterator(
        dataset=val_data,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
    )
    test_iter = BucketIterator(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
    )

    return train_iter, val_iter, test_iter, text_field, label_field, VOCAB_SIZE


def load_embeddings(emb, text_field):

    def load_embeddings1(path):
        """Load the FastText embeddings from the embedding file."""
        print("Loading pre-trained embeddings")

        embeddings = {}
        with open(path) as i:
            for line in i:
                if len(line) > 2:
                    line = line.strip().split()
                    word = line[0]
                    embedding = np.array(line[1:])
                    embeddings[word] = embedding

        return embeddings

    def initialize_embeddings(embeddings, vocabulary):
        """Use the pre-trained embeddings to initialize an embedding matrix."""
        print("Initializing embedding matrix")
        embedding_size = len(embeddings["."])
        embedding_matrix = np.zeros((len(vocabulary), embedding_size), dtype=np.float32)

        for idx, word in enumerate(vocabulary.itos):
            if word in embeddings:
                embedding_matrix[idx, :] = embeddings[word]

        return embedding_matrix

    if emb == "fasttext":

        EMBEDDING_PATH = "/Users/antoineedy/Documents/MScAI/Semester2/NLP/Coursework/code/data/cc.en.300.vec"

        # embeddings = load_embeddings1(EMBEDDING_PATH)
        # embedding_matrix = initialize_embeddings(embeddings, text_field.vocab)
        # embedding_matrix = torch.from_numpy(embedding_matrix)
        # np.save("embedding_matrix_fasttext.npy", embedding_matrix)
        embedding_matrix = np.load("data/embedding_matrix_fasttext.npy")
        embedding_matrix = torch.from_numpy(embedding_matrix)

    elif emb == "glove":

        # EMBEDDING_PATH = "data/glove.6B.300d.txt"

        # embeddings = load_embeddings1(EMBEDDING_PATH)
        # embedding_matrix = initialize_embeddings(embeddings, text_field.vocab)
        # embedding_matrix = torch.from_numpy(embedding_matrix)
        embedding_matrix = np.load("data/glove_embeddings.npy")
        embedding_matrix = torch.from_numpy(embedding_matrix)

    elif emb == "word2vec":

        model = gensim.models.KeyedVectors.load_word2vec_format(
            "data/GoogleNews-vectors-negative300.bin", binary=True
        )
        em = []
        for word in text_field.vocab.itos:
            if word in model:
                em.append(model.get_vector(word))
            else:
                em.append(np.zeros(300))
        em = np.array(em)
        embedding_matrix = torch.tensor(em, dtype=torch.float32)

    return embedding_matrix


class BiLSTMTagger(nn.Module):

    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, output_size, embeddings=None
    ):
        super(BiLSTMTagger, self).__init__()

        # 1. Embedding Layer
        if embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)

        # 2. LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=1)

        # 3. Optional dropout layer
        self.dropout_layer = nn.Dropout(p=0.5)

        # 4. Dense Layer
        self.hidden2tag = nn.Linear(2 * hidden_dim, output_size)

    def forward(self, batch_text, batch_lengths):

        embeddings = self.embeddings(batch_text)

        packed_seqs = pack_padded_sequence(embeddings, batch_lengths)
        lstm_output, _ = self.lstm(packed_seqs)
        lstm_output, _ = pad_packed_sequence(lstm_output)
        lstm_output = self.dropout_layer(lstm_output)

        logits = self.hidden2tag(lstm_output)
        return logits


def remove_predictions_for_masked_items(predicted_labels, correct_labels):

    predicted_labels_without_mask = []
    correct_labels_without_mask = []

    for p, c in zip(predicted_labels, correct_labels):
        if c > 1:
            predicted_labels_without_mask.append(p)
            correct_labels_without_mask.append(c)

    return predicted_labels_without_mask, correct_labels_without_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    train_iter,
    dev_iter,
    batch_size,
    max_epochs,
    num_batches,
    patience,
    output_path,
    label_field,
):
    NUM_CLASSES = len(label_field.vocab)
    writer = SummaryWriter()

    # add weight to indexes 3, 4, 5
    w = [0, 0, 0.0443, 0.6259, 1.0000, 0.4525]
    class_weights = torch.tensor(w).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=1
    )  # we mask the <pad> labels
    # Hinge loss
    criterion = nn.MultiMarginLoss(margin=1.0, weight=class_weights, reduction="mean")

    # optimizer = optim.Adam(model.parameters())

    # SGD
    # lr = 0.1
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # AdamW
    optimizer = optim.AdamW(model.parameters())

    train_f_score_history = []
    dev_f_score_history = []
    no_improvement = 0
    for epoch in range(max_epochs):

        total_loss = 0
        predictions, correct = [], []
        for batch in tqdm(train_iter, total=num_batches, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            text_length, cur_batch_size = batch.text[0].shape

            pred = model(batch.text[0].to(device), batch.text[1].to(device)).view(
                cur_batch_size * text_length, NUM_CLASSES
            )
            gold = batch.labels.to(device).view(cur_batch_size * text_length)

            loss = criterion(pred, gold)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, pred_indices = torch.max(pred, 1)

            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(
                batch.labels.view(cur_batch_size * text_length).numpy()
            )

            predicted_labels, correct_labels = remove_predictions_for_masked_items(
                predicted_labels, correct_labels
            )

            predictions += predicted_labels
            correct += correct_labels

        train_scores = precision_recall_fscore_support(
            correct, predictions, average="micro"
        )
        train_f_score_history.append(train_scores[2])

        print("Total training loss:", total_loss)
        print("Training performance:", train_scores)

        # tensorboard
        writer.add_scalar("train/loss", total_loss, epoch)
        writer.add_scalar("train/precision", train_scores[2], epoch)

        total_loss = 0
        predictions, correct = [], []
        for batch in dev_iter:

            text_length, cur_batch_size = batch.text[0].shape

            pred = model(batch.text[0].to(device), batch.text[1].to(device)).view(
                cur_batch_size * text_length, NUM_CLASSES
            )
            gold = batch.labels.to(device).view(cur_batch_size * text_length)
            loss = criterion(pred, gold)
            total_loss += loss.item()

            _, pred_indices = torch.max(pred, 1)
            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(
                batch.labels.view(cur_batch_size * text_length).numpy()
            )

            predicted_labels, correct_labels = remove_predictions_for_masked_items(
                predicted_labels, correct_labels
            )

            predictions += predicted_labels
            correct += correct_labels

        dev_scores = precision_recall_fscore_support(
            correct, predictions, average="micro"
        )

        print("Total development loss:", total_loss)
        print("Development performance:", dev_scores)

        writer.add_scalar("val/loss", total_loss, epoch)
        writer.add_scalar("val/precision", dev_scores[2], epoch)

        labels = label_field.vocab.itos[2:]
        labels = sorted(labels, key=lambda x: x.split("-")[-1])
        label_idxs = [label_field.vocab.stoi[l] for l in labels]

        cr = classification_report(
            correct,
            predictions,
            labels=label_idxs,
            target_names=labels,
            output_dict=True,
        )

        out = {}
        for key in cr.keys():
            if key == "accuracy":
                out[key] = cr[key]
            else:
                for new_k in ["precision", "recall", "f1-score"]:
                    out[key + "_" + new_k] = cr[key][new_k]

        for key, value in out.items():
            writer.add_scalar(f"test/{key}", value, epoch)

        t2id = ["<unk>", "<pad>", "B-O", "I-LF", "B-AC", "B-LF"]
        t2id = {i: t2id[i] for i in range(len(t2id))}

        correct = [t2id[i] for i in correct]
        predictions = [t2id[i] for i in predictions]

        # sequeval f1 score
        seqeval_f1 = seqeval_f1_score([correct], [predictions])
        print(seqeval_f1)
        writer.add_scalar("test/seqeval_f1", seqeval_f1, epoch)

        dev_f = dev_scores[2]

        dev_f = out["macro avg_f1-score"]

        if len(dev_f_score_history) > patience and dev_f < max(dev_f_score_history):
            no_improvement += 1

        elif len(dev_f_score_history) == 0 or dev_f > max(dev_f_score_history):
            print("Saving model.")
            torch.save(model, output_path)
            no_improvement = 0

        if no_improvement > patience:
            print("Macro average F1-score does not improve anymore. Stop training.")
            dev_f_score_history.append(dev_f)
            break

        dev_f_score_history.append(dev_f)

    return train_f_score_history, dev_f_score_history


def test(model, test_iter, batch_size, labels, target_names, NUM_CLASSES, toprint=True):

    total_loss = 0
    predictions, correct = [], []
    for batch in test_iter:

        text_length, cur_batch_size = batch.text[0].shape

        pred = model(batch.text[0].to(device), batch.text[1].to(device)).view(
            cur_batch_size * text_length, NUM_CLASSES
        )
        gold = batch.labels.to(device).view(cur_batch_size * text_length)

        _, pred_indices = torch.max(pred, 1)
        predicted_labels = list(pred_indices.cpu().numpy())
        correct_labels = list(batch.labels.view(cur_batch_size * text_length).numpy())

        predicted_labels, correct_labels = remove_predictions_for_masked_items(
            predicted_labels, correct_labels
        )

        predictions += predicted_labels
        correct += correct_labels
    # print(labels, target_names)
    if toprint:
        print(
            classification_report(
                correct, predictions, labels=labels, target_names=target_names
            )
        )
    return correct, predictions


TOKENIZERS_PARALLELISM = True


def results(output_path, label_field, test_iter, BATCH_SIZE, toreturn=False):
    tagger = torch.load(output_path)
    # print(tagger.eval())

    labels = label_field.vocab.itos[2:]
    labels = sorted(labels, key=lambda x: x.split("-")[-1])
    label_idxs = [label_field.vocab.stoi[l] for l in labels]

    if toreturn:
        toprint = False
    else:
        toprint = True

    c, p = test(
        tagger,
        test_iter,
        BATCH_SIZE,
        labels=label_idxs,
        target_names=labels,
        NUM_CLASSES=len(label_field.vocab),
        toprint=toprint,
    )
    labels, target_names = [4, 3, 5, 2], ["B-AC", "I-LF", "B-LF", "B-O"]

    ID2TEXT = dict()
    for i in range(len(labels)):
        ID2TEXT[labels[i]] = target_names[i]

    c = [ID2TEXT[i] for i in c]
    p = [ID2TEXT[i] for i in p]

    words = []
    for batch in test_iter:
        words += batch.text[0].tolist()

    if toreturn:
        return c, p, words

    labels = ["B-O", "B-AC", "B-LF"]
    nlabels = ["O", "Abb.", "Long-forms"]

    cm = confusion_matrix(c, p, normalize="true", labels=labels)

    plt.figure(figsize=(4, 4))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=nlabels,
        yticklabels=nlabels,
        fmt=".2f",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


### TRANSFORMER MODEL HERE


def initiate_transformer_data(datasets, model_checkpoint):
    # model_checkpoint = "antoineedy/stanford-deidentifier-base-finetuned-ner"
    batch_size = 16
    TEXT2ID = {
        "B-O": 0,
        "B-AC": 1,
        "B-LF": 2,
        "I-LF": 3,
    }

    # map the ner_tags to integers
    datasets = datasets.map(
        lambda x: {"ner_tags": [TEXT2ID[tag] for tag in x["ner_tags"]]}
    )
    # label_list = datasets["train"].features[f"{task}_tags"]
    label_list = list(set(datasets["train"]["ner_tags"][0]))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    label_all_tokens = True

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    return (
        tokenizer,
        data_collator,
        tokenized_datasets,
        model_checkpoint,
        label_list,
        batch_size,
    )


def initiate_training(
    data_collator,
    tokenized_datasets,
    model_checkpoint,
    label_list,
    batch_size,
    datasets,
    tokenizer,
):
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True
    )
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=60,
        weight_decay=0.01,
        push_to_hub=True,
    )

    metric = load_metric("seqeval")
    example = datasets["train"][0]
    tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        tp = []
        for l in true_predictions:
            tp += l
        tl = []
        for l in true_labels:
            tl += l

        cr = classification_report(tl, tp, output_dict=True)
        out = {}
        for key in cr.keys():
            if key == "accuracy":
                out[key] = cr[key]
            else:
                for new_k in ["precision", "recall", "f1-score"]:
                    out[key + "_" + new_k] = cr[key][new_k]
        return out

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss
            # w1 = [0.1, 10.0, 10.0, 10.0]
            # w2 = [22.5520,  1.5978,  1.0000,  2.2100]
            w3 = [0.0443, 0.6259, 1.0000, 0.4525]
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(w3).to(logits.device))
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # trainer.train()
    # trainer.evaluate()

    return trainer, model, tokenized_datasets, metric


def evaluate_transformer(tokenized_datasets, trainer, label_list, metric):
    predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    # print(results)
    labels = []
    for l in true_labels:
        labels += l
    pred = []
    for l in true_predictions:
        pred += l

    ID2TEXT = {0: "B-O", 1: "B-AC", 2: "B-LF", 3: "I-LF"}
    TEXT2ID = {v: k for k, v in ID2TEXT.items()}
    labels = [ID2TEXT[k] for k in labels]
    pred = [ID2TEXT[k] for k in pred]

    alabels = ["B-O", "B-AC", "B-LF"]
    nlabels = ["O", "Abb.", "Long-forms"]

    c = labels
    p = pred

    print(classification_report(c, p))

    cm = confusion_matrix(c, p, normalize="true", labels=alabels)

    plt.figure(dpi=600)
    plt.figure(figsize=(4, 4))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=nlabels,
        yticklabels=nlabels,
        fmt=".2f",
    )
    # make labels bold
    plt.ylabel("Actual", fontweight="bold")
    plt.xlabel("Predicted")
    plt.show()


### VISUALIZATION OF THE OUTPUTS HERE

import transformers


def show_visu_transformer(datasets, model_path):

    TEXT2ID = {
        "B-O": 0,
        "B-AC": 1,
        "B-LF": 2,
        "I-LF": 3,
    }
    datasets = datasets.map(
        lambda x: {"ner_tags": [TEXT2ID[tag] for tag in x["ner_tags"]]}
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if True else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=4)

    pipeline = transformers.pipeline(
        "ner", model=model, tokenizer=tokenizer, ignore_labels=[]
    )

    def choose(i=None):
        if i is None:
            i = torch.randint(0, len(datasets["test"]["tokens"]), (1,)).item()
        output = pipeline(" ".join(datasets["test"]["tokens"][i]))
        words = datasets["test"]["tokens"][i]
        truth = datasets["test"]["ner_tags"][i]

        return words, output, truth

    def choose_multiple(nb=10):
        indices = torch.randint(0, len(datasets["test"]["tokens"]), (nb,))
        indices = torch.unique(indices)
        words = []
        outputs = []
        truths = []
        for i in indices:
            w, o, t = choose(i)
            words.append(w)
            outputs.append(o)
            truths.append(t)
        return words, outputs, truths

    TEXT2ID = {
        "O": 0,
        "B-AC": 1,
        "B-LF": 2,
        "I-LF": 3,
    }

    def vizu(words, output, truth, type=None):
        sentence = " ".join(words)
        out_words = []
        out_label = []
        out_truth = []
        index = 1
        for i in range(len(output)):
            start = output[i]["start"]
            end = output[i]["end"]
            word = output[i]["word"]
            if type == 1 and "Ġ" in word:
                out_words.append(" ")
                out_label.append(0)
                index += 1
            elif type != 1 and word[0] != "#":
                out_words.append(" ")
                out_label.append(0)
                index += 1
            out_words.append(sentence[start:end])
            if type == 1:
                # print(output[i]['entity'])
                out_label.append(TEXT2ID[output[i]["entity"]])
            else:
                out_label.append(int(output[i]["entity"][-1]))
        col = {
            0: Style.RESET_ALL,
            1: Back.RED,
            2: Back.GREEN,
            3: Back.BLUE,
            4: Back.MAGENTA,
        }
        out_label = out_label[1:]
        out_words = out_words[1:]
        print("Output:  ", end="")
        for i in range(len(out_words)):
            print(col[out_label[i]], end="")
            print(out_words[i], end="")
            print(Style.RESET_ALL, end="")
        print()
        print("Truth:   ", end="")
        for i in range(len(words)):
            print(col[truth[i]], end="")
            print(words[i] + " ", end="")
            print(Style.RESET_ALL, end="")
        print()
        print()

    words, outputs, truths = choose_multiple()
    for i in range(len(words)):
        vizu(words[i], outputs[i], truths[i], type=0)


### PLOTS

import tensorboard as tb
import os
from copy import deepcopy


def convert_tb_data(root_dir, sort_by=None):
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame(
            [
                parse_tfevent(e)
                for e in summary_iterator(filepath)
                if len(e.summary.value)
            ]
        )

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ["wall_time", "name", "step", "value"]

    out = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))
    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    dict_out = dict()
    for name, group in all_df.groupby("name"):
        dict_out[name] = group.reset_index(drop=True)

    return dict_out


def all_df(path):
    out = dict()
    for folder in os.listdir(path):
        if folder[0] != ".":
            p = os.path.join(path, folder)
            out[folder] = convert_tb_data(p)
    return out


def do_it(path, isString=False):
    to_get = [
        "eval/1_precision",
        "eval/1_recall",
        "eval/1_f1-score",
        "eval/2_precision",
        "eval/2_recall",
        "eval/2_f1-score",
        "eval/weighted avg_f1-score",
    ]
    if isString:
        to_get = [
            "test/B-AC_precision",
            "test/B-AC_recall",
            "test/B-AC_f1-score",
            "test/B-LF_precision",
            "test/B-LF_recall",
            "test/B-LF_f1-score",
            "test/weighted avg_f1-score",
        ]
    all_dfs = all_df(path)
    o = []
    for key, value in all_dfs.items():
        add_to_o = []
        add_to_o.append(key)
        vs = []
        for k in to_get:
            add_to_o.append(round(value[k].tail(1).value.values[0], 3))
        o.append(add_to_o)
    out = deepcopy(o)
    for i in range(len(o)):
        for j in range(len(o[i])):
            if isinstance(o[i][j], str):
                print(o[i][j], end=" ")
            else:
                to_compare_to = [o[k][j] for k in range(len(o))]
                if o[i][j] == max(to_compare_to):
                    out[i][j] = (
                        "\\"
                        + "textbf{"
                        + str(out[i][j])
                        + "0" * (5 - len(str(out[i][j])))
                        + "}"
                    )
                else:
                    out[i][j] = str(out[i][j]) + "0" * (5 - len(str(out[i][j])))
                print("&", end=" ")
                print(out[i][j], end=" ")
        print("\\\\")


def from_tensorboard(path, isString=False):
    to_get = [
        "eval/1_precision",
        "eval/1_recall",
        "eval/1_f1-score",
        "eval/2_precision",
        "eval/2_recall",
        "eval/2_f1-score",
        "eval/weighted avg_f1-score",
    ]
    if isString:
        to_get = [
            "test/B-AC_precision",
            "test/B-AC_recall",
            "test/B-AC_f1-score",
            "test/B-LF_precision",
            "test/B-LF_recall",
            "test/B-LF_f1-score",
            "test/weighted avg_f1-score",
        ]
    all_dfs = all_df(path)
    o = []
    for key, value in all_dfs.items():
        add_to_o = []
        add_to_o.append(key)
        vs = []
        for k in to_get:
            add_to_o.append(round(value[k].tail(1).value.values[0], 3))
        o.append(add_to_o)
    out = deepcopy(o)
    df = pd.DataFrame(
        out,
        columns=[
            "Model",
            "Abb. Precision",
            "Abb. Recall",
            "Abb. F1-score",
            "LF Precision",
            "LF Recall",
            "LF F1-score",
            "Weighted Avg F1-score",
        ],
    )

    plt.figure(figsize=(6, 4))
    # plot B-AC through epochs
    for key, value in all_dfs.items():
        if isString:
            plt.plot(value["test/B-AC_f1-score"]["value"], label=key)
        else:
            plt.plot(value["eval/1_f1-score"]["value"], label=key)
    plt.title("Abbreviation F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.legend()
    plt.show()

    return df
