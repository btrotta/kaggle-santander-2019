import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# read input
train = pd.read_csv(os.path.join('input', 'train.csv'))
test = pd.read_csv(os.path.join('input', 'test.csv'))
all_data = pd.concat([train, test], axis=0)


for i in range(10):
    col = 'var_' + str(i)

    size = all_data.groupby(col)['target'].size().to_frame('size' + col)
    train.drop('size' + col, axis=1, inplace=True, errors='ignore')
    train = pd.merge(train, size, 'left', left_on=col, right_index=True)

    col_min = train[col].min()
    col_max = train[col].max()
    bins = np.arange(col_min, col_max, (col_max - col_min)/20)

    a = pd.cut(train[col], bins=bins, retbins=False, labels=False)
    plt.figure()
    for s in [1, 2, 3]:
        col_copy = train.loc[train['size' + col] == s, [col, 'target']].copy()
        col_copy.sort_values(col, inplace=True)
        rm = col_copy['target'].rolling(window=1000, min_periods=1, center=True).mean()
        plt.plot(col_copy[col].values, rm.values)
    plt.title('Smoothed mean value vs number of occurences, ' + col)
    plt.legend([1, 2, 3])
    plt.show()
