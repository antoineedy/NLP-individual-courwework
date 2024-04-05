import matplotlib.pyplot as plt
import pandas
from colorama import Back, Style
import torch


def vizu(words, output, full=False):
    # change the color of the token depending on the output
    if isinstance(output, torch.Tensor):
        output = output.squeeze().tolist()
    col = {0: Back.BLACK, 1: Back.RED, 2: Back.GREEN, 3: Back.BLUE, 4: Back.MAGENTA}
    colors = [col[i] for i in output]
    words = [word.replace("Ġ", "") for word in words]
    if not full:
        colors = colors[1:-1]
        words = words[1:-1]
    for i, word in enumerate(words):
        print(colors[i] + word, end=" ")
