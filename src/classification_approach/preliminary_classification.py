import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from classification_approach.constants import PB0, PB1
from preprocessing.wrangle import wrangle


def preliminary_classification():
    """
    :returns: x_cheap, y_cheap, x_mid, y_mid, x_expensive, y_expensive
    """
    x, y_unbucketed = wrangle()

    bins = [0, PB0, PB1, float('inf')]
    y = pd.cut(y_unbucketed, bins=bins, labels=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    x_scaler = MinMaxScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)

    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

    cheap_idxs, = np.where(y == 0)
    mid_idxs, = np.where(y == 1)
    expensive_idxs, = np.where(y == 2)

    x_cheap, y_cheap = x.iloc[cheap_idxs], y_unbucketed.iloc[cheap_idxs]
    x_mid, y_mid = x.iloc[mid_idxs], y_unbucketed.iloc[mid_idxs]
    x_expensive, y_expensive = x.iloc[expensive_idxs], y_unbucketed.iloc[expensive_idxs]

    return x_cheap, y_cheap, x_mid, y_mid, x_expensive, y_expensive
