import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from preprocessing.wrangle import wrangle

X, y = wrangle()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

X_scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

y_train = np.log(y_train)

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['poly'],
              'degree': [3, 5, 10]}

grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
pred = grid.predict(X_test)

print('Best parameters: ', grid.best_params_)
print('Best estimator: ', grid.best_estimator_)

pred = np.exp(pred)

diff = abs(pred - y_test)
print("Acceptable results percentage:", 100 * sum(diff < 100) / len(diff))
