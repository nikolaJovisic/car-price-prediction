import os

import numpy as np
import torch
import torch.optim as optim

from inference import inference_demo
from model import get_model
from dataset import ImageDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from preprocessing import preprocessing
from utils import overview

augmentation = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
)

train_transform = transforms.Compose([preprocessing, augmentation])

dataset = ImageDataset()

model_files_path = 'model_files'
torch.save(dataset.vocab, os.path.join(model_files_path, 'vocab.pt'))

train_set, validation_set, test_set = random_split(dataset, lengths=(0.65, 0.25, 0.1))

train_set.dataset.transform = train_transform
validation_set.dataset.transform = preprocessing

train_loader = DataLoader(train_set, batch_size=5, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=5, shuffle=True, drop_last=True)

print("train len:", len(train_loader))
print("test len:", len(validation_loader))

model = get_model(dataset.vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


def corrects(outputs, labels):
    with torch.no_grad():
        pred = np.argmax(outputs, axis=1)
        gt = np.argmax(labels, axis=1)
        return sum(pred == gt).item()


best_accuracy = 0.0

for epoch in range(100):
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

    validation_loss = 0.0
    validation_corrects = 0
    with torch.no_grad():
        for batch, data in enumerate(validation_loader):
            inputs, labels = data
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
            validation_corrects += corrects(outputs, labels)
            # if epoch % 10 == 0 and batch in [0, 1, 2]:
            #     overview(inputs, outputs, labels, dataset.vocab)
    validation_loss /= len(validation_loader)
    validation_corrects /= 5 * len(validation_loader)
    if validation_corrects > best_accuracy:
        print(f'New best model, accuracy update: {best_accuracy} -> {validation_corrects}')
        best_accuracy = validation_corrects
        torch.save(model.state_dict(), os.path.join(model_files_path, 'model_state_dict.pt'))

    print(
        f"[{epoch + 1}] {train_loss:.3f}, {validation_loss:.3f}, {train_corrects:.3f}, {validation_corrects:.3f}"
    )
    train_loss = 0.0

print("Finished Training")

inference_demo(model_files_path, test_set)
