import json
import os

from torch.utils.data import Dataset, DataLoader, random_split


def _read_data():
    descriptions = []
    car_prices = []
    for path in os.scandir(os.path.join("..", "..", "data", "textual")):
        with open(path) as file:
            datapoint = json.load(file)
            if datapoint["opis"] == "":
                continue
            descriptions.append(datapoint["opis"])
            car_prices.append(datapoint["cena"])
    return descriptions, car_prices


class CarDescriptionDataset(Dataset):
    def __init__(self, transform=None):
        self.descriptions, self.car_prices = _read_data()
        self.transform = transform

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        if self.transform:
            description = self.transform(description)
        car_price = self.car_prices[idx]
        return description, car_price


if __name__ == "__main__":
    dataset = CarDescriptionDataset()
    train_set, validation_set, test_set = random_split(dataset, lengths=(0.7, 0.2, 0.1))

    train_loader = DataLoader(
        dataset=train_set, batch_size=4, shuffle=True, num_workers=4, drop_last=True
    )
    validation_loader = DataLoader(
        dataset=validation_set,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=4, shuffle=True, num_workers=4, drop_last=True
    )

    print("Dataset length:", len(dataset))
    print("Training dataset length:", len(train_loader) * train_loader.batch_size)
    print(
        "Validation dataset length:",
        len(validation_loader) * validation_loader.batch_size,
    )
    print("Test dataset length:", len(test_loader) * test_loader.batch_size)
