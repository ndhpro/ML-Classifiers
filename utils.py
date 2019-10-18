import csv
import numpy as np
import sys
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2


def load_csv(file):
    f = open(file, mode='r')
    csv_reader = csv.reader(f, delimiter=',')
    X = list()
    y = list()
    b = 0
    for row in csv_reader:
        data = row[0:1024]
        label = row[1024][12:-10]
        if int(label) <= 2700:
            label = -1
            b += 1
        else:
            label = 1
        if (b <= 500 and label == -1) or label == 1:
            X.append(np.array(data).astype('float'))
            y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X = load_npz('data/data.npz')
    y = np.loadtxt('data/label.txt')

    X = StandardScaler(with_mean=False).fit_transform(X)
    X = SelectKBest(chi2, k=256).fit_transform(X, y)

    np.save('../DeepOC/data/data_256.npy', X.toarray())    
    print(X)
