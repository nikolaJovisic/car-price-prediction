import os

import numpy as np
import torch
import torch.optim as optim

from image_price.model import get_model
from image_price.dataset import ImageDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from image_price.preprocessing import preprocessing
from sklearn.metrics import r2_score

augmentation = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
)

train_transform = transforms.Compose([preprocessing, augmentation])

dataset = ImageDataset()

model_files_path = 'model_files'

train_set, validation_set, test_set, _ = random_split(dataset, lengths=(0.065, 0.025, 0.01, 0.9))

train_set.dataset.transform = preprocessing #train_transform
validation_set.dataset.transform = preprocessing

batch_size = 25

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, drop_last=True)

print("train len:", len(train_loader))
print("test len:", len(validation_loader))

model = get_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


best_accuracy = 0.0

for epoch in range(5000):
    train_loss = 0.0
    r2 = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.type(torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        with torch.no_grad():
            r2 += r2_score(labels, outputs)

    train_loss /= len(train_loader)
    r2 /= len(train_loader)

    validation_loss = 0.0
    validation_corrects = 0
    with torch.no_grad():
        # r2 = 0.0
        avg_error = 0.0
        for data in validation_loader:
            inputs, labels = data
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
            # r2 += r2_score(labels, outputs)
            avg_error += np.average(np.abs(outputs - labels))

    validation_loss /= len(validation_loader)
    # r2 /= len(validation_loader)
    avg_error /= len(validation_loader)

    # if validation_corrects > best_accuracy:
    #     print(f'New best model, accuracy update: {best_accuracy} -> {validation_corrects}')
    #     best_accuracy = validation_corrects
    #     torch.save(model.state_dict(), os.path.join(model_files_path, 'model_state_dict.pt'))

    print('epoch, train loss, validation loss, r2, avg_error')
    print(
        f"[{epoch + 1}] {train_loss:.3f}, {validation_loss:.3f}, {r2: .3f}, {avg_error: .3f}"
    )
    train_loss = 0.0

print("Finished Training")

