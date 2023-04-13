import re
from collections import Counter, OrderedDict
from torchtext.vocab import vocab


def tokenizer(text):
    return re.sub("\\W+", " ", text.lower()).split()


def text_pipeline(vocab_, text):
    return [vocab_[token] for token in tokenizer(text)]


def create_vocab(dataset):
    token_counts = Counter()
    for description, car_price in dataset:
        tokens = tokenizer(description)
        token_counts.update(tokens)
    sorted_by_freq = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq)
    vocab_ = vocab(ordered_dict)
    vocab_.insert_token("<pad>", 0)
    vocab_.insert_token("<unk>", 1)
    vocab_.set_default_index(1)
    print("Vocab size:", len(vocab_))
    return vocab_
