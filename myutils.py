import pandas as pd
import torch
import numpy as np
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


def postprocess(sentence, tokens, outputs, tokenizer):
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
            print(f"error: {o}")
            print(f"tokens: {t[i]}")
    out = [int(o[0]) for o in out]
    return sentence, out
