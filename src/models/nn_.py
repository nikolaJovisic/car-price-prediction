import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from preprocessing.wrangle import wrangle

x, y = wrangle()

# train-test split for model evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

x_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
x_val = x_scaler.transform(x_val)


x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_val = x_val.astype(np.float32)
y_train = y_train.values.astype(np.float32)
y_test = y_test.values.astype(np.float32)
y_val = y_val.values.astype(np.float32)

# Convert to 2D PyTorch tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).reshape(-1, 1)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).reshape(-1, 1)
x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(453, 500),
    nn.ReLU(),
    nn.Linear(500, 300),
    nn.ReLU(),
    nn.Linear(300, 15),
    nn.ReLU(),
    nn.Linear(15, 1),
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 30  # number of epochs to run
batch_size = 30  # size of each batch
batch_start = torch.arange(0, len(x_train), batch_size)

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
train_loss_history = []
val_loss_history = []
train_r2_history = []
val_r2_history = []

for epoch in range(n_epochs):
    print(epoch)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = x_train[start: start + batch_size]
            y_batch = y_train[start : start + batch_size]
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
    if epoch%3==0:
        model.eval()
        y_pred_val = model(x_val)
        y_pred_train = model(x_train)
        train_loss = float(loss_fn(y_pred_train, y_train))
        val_loss = float(loss_fn(y_pred_val, y_val))

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        train_r2_history.append(r2_score(y_train, y_pred_train.detach().numpy()))
        val_r2_history.append(r2_score(y_val, y_pred_val.detach().numpy()))
        print(r2_score(y_test, model(x_test).detach().numpy()))
        # plt.plot(train_loss_history, label='train_loss')
        # plt.plot(val_loss_history, label='validation_loss')
        plt.plot(train_r2_history, label='train_r2')
        plt.plot(val_r2_history, label='val_r2')
        plt.legend()
        plt.show()

    if val_loss < best_mse:
        best_pred = y_pred
        best_mse = val_loss
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)
# print("MSE: %.2f" % best_mse)
# print("RMSE: %.2f" % np.sqrt(best_mse))

train_o = model(x_train)
train_d = abs(train_o - y_train)
val_o = model(x_val)
val_d = abs(val_o - y_val)
test_o = model(x_test)
test_d = abs(test_o - y_test)



for i in zip(best_pred, y_val):
    print(i[0] - i[1])

plt.plot(train_loss_history)
plt.show()
