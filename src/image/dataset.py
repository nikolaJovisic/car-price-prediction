import json
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

def _read_data():
    ROOT = "../../data"
    IMAGES = os.path.join(ROOT, 'images')
    TEXTUAL = os.path.join(ROOT, 'textual')
    image_paths, labels = [], []
    for path in os.scandir(TEXTUAL):
        with open(path) as file:
            datapoint = json.load(file)
            _, filename = os.path.split(path)
            number = filename.split('.')[0]
            image_paths.append(os.path.join(IMAGES, number))
            labels.append(datapoint["dodatne informacije"]["boja"])
    return image_paths, labels

class ImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_paths, self.img_labels = _read_data()
        self.vocab = list(set(self.img_labels))
        self.vocab_size = len(self.vocab)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.img_paths[idx], '1.png'))
        label = self.encode(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def encode(self, word):
        one_hot = torch.zeros(self.vocab_size)
        one_hot[self.vocab.index(word)] = 1
        return one_hot

    def decode(self, one_hot):
        one_hot = torch.squeeze(one_hot)
        index = torch.where(one_hot == 1)[0]
        return self.vocab[index]
