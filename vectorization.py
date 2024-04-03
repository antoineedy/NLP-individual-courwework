from gensim.models import Word2Vec
import gensim

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class Vectorization:
    def __init__(self, method):
        self.method = method
        if method not in ["Word2Vec"]:
            raise NotImplementedError("Vectorization method not implemented")

    def fit(self, data):
        if self.method == "Word2Vec":
            pass

    def transform(self, data):
        if self.method == "Word2Vec":
            model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)
            return model
