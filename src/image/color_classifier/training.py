import torch
import torch.optim as optim
from image.color_classifier.net import ColorClassifier
from image.dataset import ImageDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, transforms

dataset = ImageDataset(
    transform=transforms.Compose(
        [Resize((32, 32)), transforms.ConvertImageDtype(torch.float)]
    )
)

train_set, test_set = random_split(dataset, lengths=(0.7, 0.3))

train_loader = DataLoader(train_set, batch_size=5, shuffle=True)
test_loader = DataLoader(test_set, batch_size=5, shuffle=True)

print('train len:', len(train_loader))
print('test len:', len(test_loader))

net = ColorClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    running_loss /= len(train_loader)
    print(f"[{epoch + 1}] train loss: {running_loss:.3f}")
    running_loss = 0.0

    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            test_loss += criterion(outputs, labels).item()
    test_loss /= len(test_loader)
    print(f"[{epoch + 1}] test loss: {test_loss:.3f}")



print("Finished Training")
