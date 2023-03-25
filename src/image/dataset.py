import json
import os
from torch.utils.data import Dataset
from torchvision.io import read_image

def _read_labels():
    labels = []
    for filename in os.scandir("../../data/textual"):
        with open(filename) as file:
            datapoint = json.load(file)
            labels.append(datapoint["cena"])
    return labels

class ImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_labels = _read_labels()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join("../../data/images", str(idx + 1), "1.png")
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
