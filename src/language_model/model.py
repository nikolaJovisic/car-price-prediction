import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)
        embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        lstm_out, (hidden, cell) = self.lstm(embeds)
        car_price = hidden[-1, :, :]
        car_price = self.fc(car_price)
        car_price = F.relu(car_price)
        return car_price
