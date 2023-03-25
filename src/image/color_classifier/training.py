import torch
import torch.optim as optim
from image.color_classifier.net import ColorClassifier
from image.dataset import ImageDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, transforms

dataset = ImageDataset(
    transform=transforms.Compose(
        [Resize((300, 300)), transforms.ConvertImageDtype(torch.float)]
    )
)

train_set, test_set = random_split(dataset, lengths=(0.7, 0.3))

train_loader = DataLoader(train_set, batch_size=5, shuffle=True)

net = ColorClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 2000 == 1999:  # print every 2000 mini-batches
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
        running_loss = 0.0

print("Finished Training")
