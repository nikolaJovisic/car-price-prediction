from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()


def show_occurrence_frequencies(df: pd.DataFrame):
    for column in df:
        if column in ['model', 'oprema', 'sigurnost', 'cena', 'snaga motora', 'kubikaza', 'kilometraza']:
            continue
        letter_counts = Counter(df[column])
        df_freq = pd.DataFrame.from_dict(letter_counts, orient='index')
        df_freq.plot(kind='bar')
        plt.title(column)
        plt.show()
