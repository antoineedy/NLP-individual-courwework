class Preprocessing:
    def __init__(self, method):
        self.method = method
        if method not in ["Nothing"]:
            raise ValueError("Preprocessing method not implemented")

    def fit(self, data):
        if self.method == "Nothing":
            pass

    def transform(self, data):
        if self.method == "Nothing":
            return data
