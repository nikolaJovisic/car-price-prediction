import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from preprocessing.wrangle import wrangle

x, y = wrangle()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

x_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
x_val = x_scaler.transform(x_val)

y_train = np.log(y_train)
y_test = np.log(y_test)
y_val = np.log(y_val)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
dval = xgb.DMatrix(x_val, label=y_val)

params = {"max_depth": 15, "eta": 0.2, "objective": "reg:squarederror", "nthread": 4}
evals = [(dtrain, "train"), (dval, "eval")]

bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=150, evals=evals, early_stopping_rounds=20)

# xgb.plot_importance(bst)
# plt.figure()
# plt.show()

results = xgb.cv(
   params=params,
   dtrain=dtrain,
   num_boost_round=150,
   nfold=5,
   early_stopping_rounds=20
)

best_rmse = results['test-rmse-mean'].min()
print("Cross validation best result:", best_rmse)

y = bst.predict(dtest)
y = np.exp(y)
y_test = np.exp(y_test)
diff = abs(y - y_test)
print("Acceptable results percentage test:", 100 * sum(diff < 100) / len(diff))

y = bst.predict(dval)
y = np.exp(y)
y_val = np.exp(y_val)
diff = abs(y - y_val)
print("Acceptable results percentage validation:", 100 * sum(diff < 100) / len(diff))

y = bst.predict(dtrain)
y = np.exp(y)
y_train = np.exp(y_train)
diff = abs(y - y_train)
print("Acceptable results percentage train:", 100 * sum(diff < 100) / len(diff))

print()
