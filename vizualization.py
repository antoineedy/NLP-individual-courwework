import matplotlib.pyplot as plt
import pandas
from colorama import Back, Style


class Vizualization:
    def __init__(self, method):
        self.method = method

    def plot(self, words, output):
        # change the color of the token depending on the output
        output = output.squeeze().tolist()
        col = {0: "black", 1: "green", 2: "red", 3: "blue", 4: "purple"}
        colors = [col[i] for i in output]
        words = [Back.RED + word for word in words]
        for i, word in enumerate(words):
            print(Style.RESET_ALL + word, end=" ")
