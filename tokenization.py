from transformers import AutoTokenizer


class Tokenization:
    def __init__(self, method):
        self.method = method
        self.tokenizer = None
        if self.method == None:
            print("Nothing to do here")
        elif self.method == "pretrained-bert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "surrey-nlp/roberta-base-finetuned-abbr"
            )
        else:
            raise ValueError("Invalid method")

    def tokenize(self, X_test):
        if self.method == None:
            return X_test
        elif self.method == "pretrained-bert":
            return self.tokenizer(
                X_test, padding=True, truncation=True, return_tensors="pt"
            )
        else:
            raise ValueError("Invalid method")
