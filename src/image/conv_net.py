import matplotlib.pyplot as plt
from torch import squeeze, transpose
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize

from image.dataset import ImageDataset

dataset = ImageDataset(transform=Resize((300, 300)))

train_set, test_set = random_split(dataset, lengths=(0.7, 0.3))

loader = DataLoader(train_set, batch_size=1, shuffle=True)
for image, label in loader:
    image = squeeze(image)
    image = transpose(image, 0, 2)
    print(label)
    plt.imshow(image)
    plt.show()
    input()

