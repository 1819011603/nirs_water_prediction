import numpy as np

from PLS.utils import loadDataSet01


def spxy(x, y, test_size=0.2):
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)  # 返回axis轴方向最大值的索引
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    return spec_train, spec_test, target_train, target_test

from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap

import numpy as np


def kennard_stone(X, y,n_samples):
    """Kennard-Stone算法从数据集X中选择n_samples个样本"""
    n_samples+=1
    # 计算样本之间的距离
    dist = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            dist[i, j] = dist[j, i] = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))

    # 随机选择一个样本作为子集的第一个元素
    subset = np.zeros(n_samples, dtype=int)
    subset[0] = np.random.choice(X.shape[0], 1)

    # 选择距离已选样本最远的样本
    for i in range(1, n_samples):
        d = np.min(dist[subset[:i], :], axis=0)
        subset[i] = np.argmax(d)
    subset.sort()
    # 获取未被选中的样本索引
    idx_remaining = np.delete(np.arange(X.shape[0]), subset)

    return X[subset], X[idx_remaining],y[subset], y[idx_remaining]


if __name__ == '__main__':
    x,y = loadDataSet01("total.txt")
    arr = np.arange(len(y))

    # Shuffle the array with random seed 3
    np.random.seed(3)
    np.random.shuffle(arr)
    x = x[arr]
    y = y[arr]



    # spec_train, spec_test, target_train, target_test = spxy(x,y.ravel(),int(len(x)* 0.8))
    spec_train, spec_test, target_train, target_test = kennard_stone(x,y.ravel(),int(len(x)* 0.8))
    target_train = target_train.reshape(-1,1)
    target_test = target_test.reshape(-1,1)
    train = np.concatenate((spec_train, target_train), axis=1)
    test = np.concatenate((spec_test, target_test), axis=1)
    np.savetxt(fname='train_ks.txt', X=train, delimiter=', ', newline='\n',fmt='%.4f')
    np.savetxt(fname='test_ks.txt', X=test, delimiter=', ', newline='\n',fmt='%.4f')