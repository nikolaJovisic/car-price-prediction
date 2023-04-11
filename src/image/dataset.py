import json
import os

from torch.utils.data import Dataset
from torchvision.io import read_image

from image.utils import encode_one_hot, decode_one_hot


def _read_data():
    ROOT = os.path.join(os.path.dirname(__file__), "../../data")
    IMAGES = os.path.join(ROOT, "images")
    TEXTUAL = os.path.join(ROOT, "textual")
    image_paths, labels = [], []
    for path in os.scandir(TEXTUAL):
        with open(path) as file:
            datapoint = json.load(file)
            _, filename = os.path.split(path)
            number = filename.split(".")[0]
            color = datapoint["dodatne informacije"]["boja"]
            for image_path in os.scandir(os.path.join(IMAGES, number)):
                image_paths.append(image_path.path)
                labels.append(color)
            # if color in ["srebrna", "siva"]:
            #     color = "bela"
            # if color in ["braon"]:
            #     color = "crna"
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
        image = read_image(self.img_paths[idx])
        label = self.encode(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def encode(self, word):
        return encode_one_hot(self.vocab, word)

    def decode(self, one_hot):
        return decode_one_hot(self.vocab, one_hot)
