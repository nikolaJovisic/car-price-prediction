import math

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import hsv

from image.color_classifier.model import ColorClassifier, get_model
from image.dataset import ImageDataset
from torch import nn, squeeze, transpose
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, transforms, CenterCrop, Lambda
from kornia.color.hsv import RgbToHsv


def scale_hue(image):
    """
    RgbToHsv in kornia package scales hue to [0, 2pi],
    here is scaled to [0, 1].
    """
    image[0] /= 2 * math.pi
    return image


preprocessing = transforms.Compose(
    [
        Resize((32, 32)),
        transforms.ConvertImageDtype(torch.float),
        RgbToHsv(),
        Lambda(lambd=scale_hue),
    ]
)

augmentation = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
)

train_transform = transforms.Compose([preprocessing, augmentation])

dataset = ImageDataset()

train_set, test_set = random_split(dataset, lengths=(0.7, 0.3))

train_set.dataset.transform = train_transform
test_set.dataset.transform = preprocessing

train_loader = DataLoader(train_set, batch_size=5, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=5, shuffle=True, drop_last=True)

print("train len:", len(train_loader))
print("test len:", len(test_loader))

model = get_model(dataset.vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


def corrects(outputs, labels):
    with torch.no_grad():
        pred = np.argmax(outputs, axis=1)
        gt = np.argmax(labels, axis=1)
        return sum(pred == gt).item()


def overview(inputs, outputs, labels):
    for image, output, label in zip(inputs, outputs, labels):
        image_disp = squeeze(image)
        image_disp = transpose(image_disp, 0, 2)
        image_disp = hsv_to_rgb(image_disp)
        plt.imshow(image_disp)
        plt.show()
        print("GT: ", dataset.decode(label))
        print("Prediction: ", dataset.decode(output == np.max(output.numpy())))
        input()


for epoch in range(200):
    train_loss = 0.0
    train_corrects = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_corrects += corrects(outputs, labels)
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_corrects /= 5 * len(train_loader)

    test_loss = 0.0
    test_corrects = 0
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            inputs, labels = data
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            test_corrects += corrects(outputs, labels)
            # if epoch % 10 == 0 and batch in [0, 1, 2]:
            #     overview(inputs, outputs, labels)
    test_loss /= len(test_loader)
    test_corrects /= 5 * len(test_loader)

    print(
        f"[{epoch + 1}] {train_loss:.3f}, {test_loss:.3f}, {train_corrects:.3f}, {test_corrects:.3f}"
    )
    train_loss = 0.0


print("Finished Training")
