import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from preprocessing.wrangle import wrangle

x, y = wrangle()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

x_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

y_train = np.log(y_train)


model = SVR(C=50.0, degree=10)
model.fit(x_train, y_train)
pred = model.predict(x_train)


pred = np.exp(pred)

diff = abs(pred - y_train)
print("Acceptable results percentage:", 100 * sum(diff < 100) / len(diff))
