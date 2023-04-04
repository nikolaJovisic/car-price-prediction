import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from torch import squeeze, transpose


def encode_one_hot(vocab, word):
    one_hot = torch.zeros(len(vocab))
    one_hot[vocab.index(word)] = 1
    return one_hot


def decode_one_hot(vocab, one_hot):
    one_hot = torch.squeeze(one_hot)
    index = torch.where(one_hot == 1)[0]
    return vocab[index]


def overview(inputs, outputs, labels, vocab):
    with torch.no_grad():
        for image, output, label in zip(inputs, outputs, labels):
            image_disp = squeeze(image)
            image_disp = transpose(image_disp, 0, 2)
            image_disp = hsv_to_rgb(image_disp)
            plt.imshow(image_disp)
            plt.show()
            print("GT: ", decode_one_hot(vocab, label))
            print(
                "Prediction: ", decode_one_hot(vocab, output == np.max(output.numpy()))
            )
            input()
