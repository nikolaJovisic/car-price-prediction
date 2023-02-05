import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.wrangle import wrangle
import xgboost as xgb

x, y, scaler = wrangle()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

param = {'max_depth': 150, 'eta': 1, 'objective': 'reg:squarederror'}
param['nthread'] = 4
evallist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(param, dtrain, 10, evallist)
y = bst.predict(dtest)


y_exp = np.expand_dims(y, axis=0).transpose([1, 0])
c = np.concatenate((x_test, y_exp), axis=1)

r = scaler.inverse_transform(c)
r = r[:, -1]

y_org = np.expand_dims(y_test, axis=0).transpose([1, 0])
c_org = np.concatenate((x_test, y_org), axis=1)

r_org = scaler.inverse_transform(c_org)
r_org = r_org[:, -1]

diff = abs(r) - abs(r_org)
print('Acceptable results percentage:', 100*sum(diff < 200)/len(diff))

print()