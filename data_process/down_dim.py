import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


Data_path = '../Datas/Datas.csv'
label_path = '../Datas/labels.csv'


def load_data(Data_path, label_path) -> 'ndarry':
    pd_data = pd.read_csv(Data_path)
    pd_label = pd.read_csv(label_path)

    X = np.array(pd_data)
    y = np.array(pd_label)

    return X,y

if __name__ == '__main__':
    X,y = load_data(Data_path,label_path)
    mds = manifold.MDS(n_components=10)
    X_r = mds.fit_transform(X)
    print(X_r)
    print(y)