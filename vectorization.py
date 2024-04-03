import gensim.models
import gensim
import numpy as np
import torch

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class Vectorization:
    def __init__(self, method):
        self.model = None
        self.method = method
        if method not in ["Word2Vec"]:
            raise NotImplementedError("Vectorization method not implemented")

    def fit(self, data):
        if self.method == "Word2Vec":
            self.model = gensim.models.Word2Vec(
                sentences=data, vector_size=100, window=5, min_count=1
            )

    def transform(self, data, y):
        new_y = []
        for t in y:
            if len(t) > 50:
                new_y.append(t[:50])
            else:
                new_y.append(t + [0] * (50 - len(t)))
        y = new_y
        if self.method == "Word2Vec":
            # max size = 50
            out = []
            for i, sentence in enumerate(data):
                temp = []
                for word in sentence:
                    if word in self.model.wv.index_to_key:
                        temp.append(self.model.wv[word])
                    else:
                        temp.append(list(np.zeros(100)))
                    if len(temp) == 50:
                        break
                if len(temp) < 50:
                    temp += [list(np.zeros(100))] * (50 - len(temp))
                if len(y[i]) < 50:
                    y[i] += [0] * (50 - len(y[i]))
                else:
                    y[i] = y[i][:50]
                out.append(temp)
            y_bis = np.zeros_like(y)
            y_bis = y_bis.tolist()
            for i in range(len(y)):
                for j in range(len(y[i])):
                    temp = [0, 0, 0, 0, 0]
                    temp[y[i][j]] = 1
                    y_bis[i][j] = temp

            y = y_bis

            return torch.tensor(np.array(out), dtype=torch.float32), torch.tensor(
                np.array(y), dtype=torch.long
            )
