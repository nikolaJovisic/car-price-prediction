import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from ..preprocessing.wrangle import wrangle

X, y = wrangle()

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

x_scaler = preprocessing.MinMaxScaler().fit(X_train)
x_train = x_scaler.transform(X_train)
x_test = x_scaler.transform(X_test)

X_train = X_train.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)

# Convert to 2D PyTorch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).reshape(-1, 1)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(226, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100  # number of epochs to run
batch_size = 30  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    print(epoch)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_pred = y_pred
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

for i in zip(best_pred, y_test):
    print(i[0] - i[1])

plt.plot(history)
plt.show()
