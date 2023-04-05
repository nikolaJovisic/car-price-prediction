import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from classification_approach.constants import PB0, PB1, BS, PB2
from classification_approach.preliminary_classification import preliminary_classification

_, _, _, _, x, y = preliminary_classification()

bins = [*range(PB1, PB2 + BS, BS)]
y = pd.cut(y, bins=bins, labels=False)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

x_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

model = SVC(C=50.0, degree=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# _, y_pred = y_pred.max(dim=1)
# y_true = y_test.argmax(dim=1)
report = classification_report(y_test, y_pred)
print(report)


