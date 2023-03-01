from typing import Collection

import numpy as np
import pandas as pd
import prince


def _one_hot(df: pd.DataFrame, columns: Collection[str]):
    """
    Performs one-hot encoding on a collection of columns.
    :param df: Dataframe to get columns from.
    :param columns: Columns to use.
    :return: Matrix of one-hot encoded atributes.
    """

    encodings = []
    for column in columns:
        encodings.append(pd.get_dummies(df[column]))
    return pd.concat(encodings, axis="columns")


def _fit(df: pd.DataFrame):
    """
    Returns MCA model fitted on dataframe with rank number of components.
    :param df: Dataframe to fit.
    :return: Fitted MCA model.
    """
    rank = np.linalg.matrix_rank(df)
    mca = prince.MCA(n_components=rank-1)
    return mca.fit(df)


def encode_columns(df: pd.DataFrame, columns: Collection[str]):
    """
    Encodes given categorical attribute columns to minimal-dimensional
    linear transformation of their one-hot encodings.
    :param df: Dataframe containing columns.
    :param columns: Columns of dataframe to be encoded.
    :return: Dataframe with new columns for encoded attributes instead of old columns.
    """
    encoded = _one_hot(df, columns)
    mca = _fit(encoded)
    transformed = mca.transform(encoded)
    df.drop(columns=columns, inplace=True)
    return df.join(transformed)


def save(model: prince.MCA, path: str):
    """
    Saves MCA model.
    """
    pass


def load(path: str):
    """
    Loads MCA model.
    """
    pass
