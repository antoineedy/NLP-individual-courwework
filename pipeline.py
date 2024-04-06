from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from colorama import Back, Style


class Pipeline:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def get_results(self):
        self.X = self.tokenizer(
            list(self.dataset["sentences"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        all_words, all_outs, all_y = self.__test(
            self.model, self.tokenizer, self.X, self.dataset
        )
        self.all_words = all_words
        self.all_outs = all_outs
        self.all_y = all_y
        return all_words, all_outs, all_y

    def get_confusion_matrix(self):
        TEXT2ID = {
            "B-O": 0,
            "B-AC": 1,
            "B-LF": 3,
            "I-LF": 4,
        }
        ID2TEXT = {v: k for k, v in TEXT2ID.items()}
        cm = confusion_matrix(self.all_y, self.all_outs, normalize="true")
        # plot
        plt.figure(figsize=(5, 5))
        sns.heatmap(
            cm,
            annot=True,
            xticklabels=ID2TEXT.values(),
            yticklabels=ID2TEXT.values(),
            cmap="Blues",
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def vizu(self, start, end):
        words = self.all_words[start:end]
        output = self.all_outs[start:end]
        truth = self.all_y[start:end]
        if isinstance(output, torch.Tensor):
            output = output.squeeze().tolist()
        col = {0: Back.BLACK, 1: Back.RED, 2: Back.GREEN, 3: Back.BLUE, 4: Back.MAGENTA}
        colors1 = [col[i] for i in output]
        colors2 = [col[i] for i in truth]
        words = [word.replace("Ġ", "") for word in words]
        print(Style.RESET_ALL + "Output:")
        for i, word in enumerate(words):
            print(colors1[i] + word, end=" ")
        print(Style.RESET_ALL + "\nTruth:")
        for i, word in enumerate(words):
            print(colors2[i] + word, end=" ")

    def __test(self, model, tokenizer, X, df):
        model.eval()
        outputs = model(**X)
        all_words = []
        all_outs = []
        all_y = []
        for i in range(len(df)):
            words = [tokenizer.decode(token) for token in X["input_ids"][i]]
            preds = outputs["logits"][i].max(1).indices
            sentence, out = self.__postprocess(
                df["sentences"][i], words, preds, tokenizer
            )
            all_words += sentence
            all_outs += out
            all_y += list(df["ids"][i])
        return all_words, all_outs, all_y

    def __postprocess(self, sentence, tokens, outputs, tokenizer):
        tokens = [token for token in tokens if token[0] != "<"]
        outputs = outputs[1:]
        sentence = sentence.split()
        l = []
        for word in sentence:
            token = tokenizer.tokenize(word)
            l.append(len(token))
        a = 0
        out = []
        t = []
        for i in range(len(l)):
            out.append(outputs[a : a + l[i]])
            t.append(tokens[a : a + l[i]])
            a += l[i]
        for i, o in enumerate(out):
            if not all(x == o[0] for x in o):
                # print(f"error: {o}")
                # print(f"tokens: {t[i]}")
                pass
        out = [int(o[0]) for o in out]
        return sentence, out
