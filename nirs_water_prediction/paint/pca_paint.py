import pathlib

import torch
import torch.nn as nn
import torch.optim as optim

from numpy import mat, zeros
from scipy import interpolate
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import nirs.util_paint


def loadDataSet01(filename, Separator=', '):
    fr = open(filename)
    arrayLines = fr.readlines()
    assert len(arrayLines) != 0
    num = len(arrayLines[0].split(Separator)) - 1
    row = len(arrayLines)
    x = mat(zeros((row, num)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split(Separator)
        curLine = [float(i) for i in curLine]
        x[index, :] = curLine[0: -1]
        # y[index, :] = curLine[-1]/100
        y[index, :] = curLine[-1]
        index += 1
    fr.close()
    return np.array(x), np.array(y).ravel()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nirs.nirs_processing import sg,dt,snv
# 读取数据
from  nirs.parameters import X_train_copy,X_test
# 主成分分析
pca = PCA()
data = np.concatenate((X_train_copy,X_test),axis=0)
# data = dt(sg(data))

pca.fit(data)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)


def nihe(x,y,num=1024):
    f = interpolate.interp1d(x, y, kind='cubic')

    # 构造插值后的新数据点
    x_new = np.linspace(x[0], x[-1], num=num, endpoint=True)
    y_new = f(x_new)
    return y_new

# 获取每个特征的波长
wavelengths = np.arange(data.shape[1])

# 获取PC1、PC2和PC3的载荷值
seq = 17

pc1_loadings = loadings[:, 0]
pc1_loadings = nihe(wavelengths[::seq],pc1_loadings[::seq])

pc2_loadings = loadings[:, 1]
pc2_loadings = nihe(wavelengths[::seq],pc2_loadings[::seq])
pc3_loadings = loadings[:, 2]
pc3_loadings = nihe(wavelengths[::seq],pc3_loadings[::seq])


selected_wavelengths = []
# 根据PC1、PC2和PC3的正值波峰和负值波谷选择特征波长
max_indices = [i for i in range(1, len(pc1_loadings)-1) if pc1_loadings[i-1] < pc1_loadings[i] > pc1_loadings[i+1]]
min_indices = [i for i in range(1, len(pc1_loadings)-1) if pc1_loadings[i-1] > pc1_loadings[i] < pc1_loadings[i+1]]
selected_wavelengths1 = max_indices + min_indices
selected_wavelengths1.sort()
selected_wavelengths.extend(selected_wavelengths1)
max_indices = [i for i in range(1, len(pc2_loadings)-1) if pc2_loadings[i-1] < pc2_loadings[i] > pc2_loadings[i+1]]
min_indices = [i for i in range(1, len(pc2_loadings)-1) if pc2_loadings[i-1] > pc2_loadings[i] < pc2_loadings[i+1]]
selected_wavelengths2 = max_indices + min_indices
selected_wavelengths2.sort()
selected_wavelengths.extend(selected_wavelengths2)
max_indices = [i for i in range(1, len(pc3_loadings)-1) if pc3_loadings[i-1] < pc3_loadings[i] > pc3_loadings[i+1]]
min_indices = [i for i in range(1, len(pc3_loadings)-1) if pc3_loadings[i-1] > pc3_loadings[i] < pc3_loadings[i+1]]
selected_wavelengths3 = max_indices + min_indices
selected_wavelengths.extend(selected_wavelengths3)
selected_wavelengths.sort()
print(selected_wavelengths)
selected_wavelengths = np.array(selected_wavelengths)
wavelengths = np.linspace(wavelengths[0], wavelengths[-1], num=len(pc1_loadings), endpoint=True)
# 绘制载荷图
# fig, ax = plt.subplots(figsize=(16, 12),dpi=300)
# from nirs.parameters import xpoints
# ax = plt.subplot(2, 1, 1)
# ax.plot(wavelengths, pc1_loadings, label='PC1')
# ax.plot(wavelengths, pc2_loadings, label='PC2')
# ax.plot(wavelengths, pc3_loadings, label='PC3')
# ax.scatter(wavelengths[selected_wavelengths1], pc1_loadings[selected_wavelengths1], marker='s', s=100, color='red',label="Characteristic wavelength")
# ax.scatter(wavelengths[selected_wavelengths2], pc2_loadings[selected_wavelengths2], marker='s', s=100, color='red')
# ax.scatter(wavelengths[selected_wavelengths3], pc3_loadings[selected_wavelengths3], marker='s', s=100, color='red')
# plt.xticks(np.arange(256)[::20],xpoints[::20])
# ax.set_xlabel('Wavelengths(nm)')
# ax.set_ylabel('PCA Loading Value')
# font = FontProperties()
# font.set_weight('bold')
# font.set_size(16)
# font.set_family('Arial')
#
# # 设置标题
# title = '(a)'
# ax.set_title(title, fontproperties=font, y=-0.2)
#
# ax.legend()
#
# def nihe1(x,y,num=1024):
#     f = interpolate.interp1d(x, y, kind='cubic')
#
#     # 构造插值后的新数据点
#     x_new = np.linspace(x[0], x[-1], num=num, endpoint=True)
#     y_new = f(x_new)
#     return x_new,y_new
#
# ax = plt.subplot(2, 1, 2)
# a,b = nihe1(np.arange(256)[::seq],data.mean(axis=0)[::seq])
# ax.plot(a,b, label='Average spectra')
# m =  data.mean(axis=0)
# ax.scatter(a[(selected_wavelengths).astype(int)],b[(selected_wavelengths).astype(int)], marker='s', s=100, color='red',label="Characteristic wavelength")
# # 设置标题
# ax.set_xlabel('Wavelengths(nm)')
# ax.set_ylabel('Reflectance')
#
# plt.xticks(np.arange(256)[::20],xpoints[::20])
# result = xpoints[(selected_wavelengths/4).astype(int)]
# plt.legend()
# result.sort()
# print(*result,sep=",")
#
# title = '(b)'
# ax.set_title(title, fontproperties=font, y=-0.2)
# from utils import get_log_name
# plt.tight_layout()
# name = get_log_name("pca", "jpeg", "./pca")
# print("save picture in {}".format(name))
# plt.savefig(name, dpi=300)
# plt.show()


selected_wavelengths = np.array([12, 99, 105, 210, 226, 316, 400, 423, 445, 468, 476, 537, 545, 579, 612, 675, 697, 707, 845, 873, 990])

fig, ax = plt.subplots(figsize=(9, 6),dpi=100)
from nirs.parameters import xpoints, X_train

# ax = plt.subplot(2, 1, 1)
ax.plot(wavelengths, pc1_loadings, label='PC1')
ax.plot(wavelengths, pc2_loadings, label='PC2')
ax.plot(wavelengths, pc3_loadings, label='PC3')
ax.scatter(wavelengths[selected_wavelengths1], pc1_loadings[selected_wavelengths1], marker='s', s=100,  color='none',edgecolor="red",label="Characteristic wavelength")
ax.scatter(wavelengths[selected_wavelengths2], pc2_loadings[selected_wavelengths2], marker='s', s=100,  color='none',edgecolor="red")
ax.scatter(wavelengths[selected_wavelengths3], pc3_loadings[selected_wavelengths3], marker='s', s=100,  color='none',edgecolor="red")
plt.xticks(np.arange(256)[::51],xpoints[::51])
ax.set_xlabel('Wavelengths(nm)')
ax.set_ylabel('PCA Loading Value')
plt.ylim(-0.05,0.15)

# 设置标题
title = '(a)'
# ax.set_title(title, fontproperties=font, y=-0.2)

ax.legend(loc="upper right", ncol=2)
from utils import get_log_name

suff = 'pdf'
picture_name = "pca"
dir_path = "./pdf"
picture_path = get_log_name(picture_name, suff=suff, dir_path=dir_path)

print("save in {}".format(picture_path))
plt.savefig(picture_path, format='pdf')
plt.show()

def nihe1(x,y,num=1024):
    f = interpolate.interp1d(x, y, kind='cubic')

    # 构造插值后的新数据点
    x_new = np.linspace(x[0], x[-1], num=num, endpoint=True)
    y_new = f(x_new)
    return x_new,y_new
fig, ax = plt.subplots(figsize=(9, 6),dpi=100)
# ax = plt.subplot(2, 1, 2)
a,b = nihe1(np.arange(256)[::seq],data.mean(axis=0)[::seq])
ax.plot(a,b, label='Average spectra')
m =  data.mean(axis=0)
ax.scatter(a[(selected_wavelengths).astype(int)],b[(selected_wavelengths).astype(int)], marker='s', s=100, color='none',edgecolor="red",label="Characteristic wavelength")
# 设置标题
ax.set_xlabel('Wavelengths(nm)')
ax.set_ylabel('Reflectance')

plt.xticks(np.arange(256)[::51],xpoints[::51])
result = xpoints[(selected_wavelengths/4).astype(int)]
plt.legend()
result.sort()
print(*result,sep="、")

title = '(b)'
# ax.set_title(title, fontproperties=font, y=-0.2)
from utils import get_log_name

suff = 'pdf'
picture_name = "pca"
dir_path = "./pdf"
picture_path = get_log_name(picture_name, suff=suff, dir_path=dir_path)

print("save in {}".format(picture_path))
plt.savefig(picture_path, format='pdf')

plt.show()
