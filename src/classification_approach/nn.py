import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from classification_approach.bounds import B0
from classification_approach.preliminary_classification import preliminary_classification

x, y, _, _, _, _ = preliminary_classification()

bins = [*range(0, B0 + 1000, 1000)]
y = pd.cut(y, bins=bins, labels=False)
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

x_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

model = nn.Sequential(
    nn.Linear(226, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 9),
)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 250
batch_size = 30
batch_start = torch.arange(0, len(x_train), batch_size)

best_loss = np.inf
best_weights = None
history = []

for epoch in range(n_epochs):
    print(epoch)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = x_train[start: start + batch_size]
            y_batch = y_train[start : start + batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_postfix(mse=float(loss))
    model.eval()
    y_pred = model(x_test)
    loss_val = loss_fn(y_pred, y_test)
    loss_val = float(loss_val)
    history.append(loss_val)
    if loss_val < best_loss:
        best_pred = y_pred
        best_loss = loss_val
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)
print("CrossEntropyLoss: %.2f" % best_loss)

y_pred = model(x_test)

_, y_pred = y_pred.max(dim=1)
y_true = y_test.argmax(dim=1)
conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)


plt.plot(history)
plt.show()
