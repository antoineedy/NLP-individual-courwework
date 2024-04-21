from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchtext.data import Field, Dataset, Example, BucketIterator
import gensim

import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, classification_report
from seqeval.metrics import f1_score as seqeval_f1_score

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import seaborn as sns

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

    def load_embeddings(path):
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

        embeddings = load_embeddings(EMBEDDING_PATH)
        embedding_matrix = initialize_embeddings(embeddings, text_field.vocab)
        embedding_matrix = torch.from_numpy(embedding_matrix)
        print(embedding_matrix.shape)

    elif emb == "glove":

        EMBEDDING_PATH = "data/glove.6B.300d.txt"

        embeddings = load_embeddings(EMBEDDING_PATH)
        embedding_matrix = initialize_embeddings(embeddings, text_field.vocab)
        embedding_matrix = torch.from_numpy(embedding_matrix)
        print(embedding_matrix.shape)

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
        print(embedding_matrix.shape)

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
    # criterion = nn.MultiMarginLoss(margin=1.0, weight=class_weights, reduction='mean')

    optimizer = optim.Adam(model.parameters())

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


def test(model, test_iter, batch_size, labels, target_names, NUM_CLASSES):

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
    print(labels, target_names)
    print(
        classification_report(
            correct, predictions, labels=labels, target_names=target_names
        )
    )
    return correct, predictions


def results(output_path, label_field, test_iter, BATCH_SIZE):
    tagger = torch.load(output_path)
    print(tagger.eval())

    labels = label_field.vocab.itos[2:]
    labels = sorted(labels, key=lambda x: x.split("-")[-1])
    label_idxs = [label_field.vocab.stoi[l] for l in labels]

    c, p = test(
        tagger,
        test_iter,
        BATCH_SIZE,
        labels=label_idxs,
        target_names=labels,
        NUM_CLASSES=len(label_field.vocab),
    )
    labels, target_names = [4, 3, 5, 2], ["B-AC", "I-LF", "B-LF", "B-O"]

    ID2TEXT = dict()
    for i in range(len(labels)):
        ID2TEXT[labels[i]] = target_names[i]

    c = [ID2TEXT[i] for i in c]
    p = [ID2TEXT[i] for i in p]

    labels = ["B-O", "B-AC", "B-LF"]
    nlabels = ["O", "Abb.", "Long-forms"]

    cm = confusion_matrix(c, p, normalize="true", labels=labels)

    plt.rcParams["font.family"] = "serif"

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=nlabels,
        yticklabels=nlabels,
        fmt=".2f",
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
