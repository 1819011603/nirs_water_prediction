import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nirs.parameters import *
from  nirs.nirs_processing import sg,dt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
def nihe(x,y,num=1000):
    f = interpolate.interp1d(x, y, kind='cubic')

    # 构造插值后的新数据点
    x_new = np.linspace(x[0], x[-1], num=num, endpoint=True)
    y_new = f(x_new)
    return y_new
# 计算主成分载荷
def calc_pc_loadings(data):
    # 对每一列进行标准化
    # data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    # 计算协方差矩阵
    pca = PCA(n_components=3)


    pca.fit(data)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return   loadings[:, :3]

# 绘制主成分载荷图
def plot_pc_loadings(data):
    # 计算主成分载荷
    pc_loadings = calc_pc_loadings(data)
    # 绘制主成分载荷图
    fig, ax = plt.subplots()

    x =  np.linspace(0,256, num=1000, endpoint=True)
    pc1 = nihe(np.arange(256)[::8], pc_loadings[:, 0][::8])
    pc2 = nihe(np.arange(256)[::8], pc_loadings[:, 1][::8])
    pc3 = nihe(np.arange(256)[::8], pc_loadings[:, 2][::8])
    ax.plot(x,nihe(np.arange(256)[::8], pc_loadings[:, 0][::8]), label='PC1')
    ax.plot(x, nihe(np.arange(256)[::8], pc_loadings[:, 1][::8]), label='PC2')
    ax.plot(x, nihe(np.arange(256)[::8], pc_loadings[:, 2][::8]), label='PC3')
    # 找出正值波峰和负值波谷
    max_idx = argrelextrema(pc_loadings[:, :3], np.greater)[0]
    min_idx = argrelextrema(pc_loadings[:, :3], np.less)[0]
    max_wavelengths = x[max_idx]
    min_wavelengths = x[min_idx]
    # 绘制正值波峰和负值波谷
    # ax.plot(max_wavelengths, pc_loadings[:, :3][max_idx], 'rs', mfc='none', label='Max')
    # ax.plot(min_wavelengths, pc_loadings[:, :3][min_idx], 'bs', mfc='none', label='Min')
    ax.legend()
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Loadings')
    ax.set_title('PC Loadings')
    plt.show()

# 示例代码
# data = dt(sg(X_train))
data = X_train
plot_pc_loadings(data)
