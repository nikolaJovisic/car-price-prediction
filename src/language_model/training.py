from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn

from language_model.dataset import CarDescriptionDataset
from language_model.model import LSTM
from language_model.preprocessing import text_pipeline, create_vocab

from datetime import datetime
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _text, _label in batch:
        label_list.append(float(_label))
        processed_text = torch.tensor(text_pipeline(vocab_, _text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    padded_text_list, label_list, lengths = (
        padded_text_list.to(device),
        label_list.to(device),
        lengths.to(device),
    )
    return padded_text_list, label_list, lengths


def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    dataset_length = len(dataloader) * train_loader.batch_size
    current_batch = 0
    for text_batch, label_batch, lengths in dataloader:
        current_batch += batch_size
        if current_batch % 100 == 0:
            print("Batches:", current_batch, "/", dataset_length)
        # for text in text_batch:
        #     print(' '.join(vocab_.get_itos()[t] for t in text))
        optimizer.zero_grad()
        if isinstance(model, BertForSequenceClassification):
            if text_batch.shape[1] > 512:
                text_batch = torch.stack([text[:512] for text in text_batch])
            pred = model(text_batch).logits[:, 0]
        else:
            pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += (abs(pred - label_batch) < 100).sum().item()
        total_loss += loss.item() * label_batch.size(0)
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            if isinstance(model, BertForSequenceClassification):
                if text_batch.shape[1] > 512:
                    text_batch = torch.stack([text[:512] for text in text_batch])
                pred = model(text_batch).logits[:, 0]
            else:
                pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += (abs(pred - label_batch) < 100).sum().item()
            total_loss += loss.item() * label_batch.size(0)
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


dataset = CarDescriptionDataset()
vocab_ = create_vocab(dataset)

# Hyper parameters
vocab_size = len(vocab_)
embed_dim = 10
rnn_hidden_size = 16
fc_hidden_size = 16
num_epochs = 1000
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Dataset split and loading
train_dataset, validation_dataset, test_dataset = random_split(
    dataset, lengths=(0.7, 0.2, 0.1)
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)
validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)

# Create model
# model = LSTM(embedding_dim=embed_dim, hidden_dim=rnn_hidden_size, vocab_size=vocab_size)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
print("Model:", model)

# Train
for epoch in range(num_epochs):
    t1 = datetime.now()
    acc_train, loss_train = train(train_loader)
    acc_val, loss_val = evaluate(validation_loader)
    t2 = datetime.now()
    print(f"ETA {(t2-t1).total_seconds()} s")
    print(f"Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {acc_val:.4f}")
    print(f"              loss: {loss_train:.4f} val_loss: {loss_val:.4f}")

# # Printing
# text_batch, label_batch, length_batch = next(iter(train_loader))
# print(text_pipeline(vocab_, 'a sad auto asd'))
# print(text_batch)
# print(label_batch)
# print(length_batch)
# print(text_batch.shape)
