import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from src.preprocessing.wrangle import wrangle
import xgboost as xgb

df = wrangle()

x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

x_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

y_train = np.log(y_train)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

param = {'max_depth': 150, 'eta': 1, 'objective': 'reg:squarederror'}
param['nthread'] = 4
evallist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(param, dtrain, 10, evallist)
y = bst.predict(dtest)
y = np.exp(y)

diff = abs(y - y_test)
print('Acceptable results percentage:', 100*sum(diff < 100)/len(diff))

print()