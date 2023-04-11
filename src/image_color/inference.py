import os

import torch
from torch.utils.data import DataLoader

from image_color.model import get_model
from image_color.preprocessing import preprocessing
from image_color.utils import overview


def inference_demo(path, test_dataset):
    vocab = torch.load(os.path.join(path, 'vocab.pt'))
    vocab_size = len(vocab)

    model = get_model(vocab_size)
    model.load_state_dict(torch.load(os.path.join(path, 'model_state_dict.pt')))
    model.eval()

    test_dataset.dataset.transform = preprocessing

    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, drop_last=True)

    for batch, data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs)
        overview(inputs, outputs, labels, vocab)


