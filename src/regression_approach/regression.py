import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from preprocessing.wrangle import wrangle

X, y = wrangle()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

X_scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

y_train = np.log(y_train)

# model = Lasso(alpha=0.1)
# model = LinearRegression()
model = DecisionTreeRegressor()
# model = KNeighborsRegressor(n_neighbors=6)
# model = RandomForestRegressor()
# model = XGBRegressor()

model.fit(X_train, y_train)

pred = model.predict(X_test)
pred = np.exp(pred)

diff = abs(pred - y_test)
print("Acceptable results percentage:", 100 * sum(diff < 100) / len(diff))
