import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.font_manager import FontProperties
from scipy import interpolate
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nirs.parameters import *
from  nirs.nirs_processing import sg,dt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

X = X_train
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # 定义SNE模型
# sne = TSNE(n_components=2, perplexity=30.0, random_state=0)
#
# # 训练SNE模型
# X_sne = sne.fit_transform(X)
#
# # 绘制SNE可视化图像
# plt.scatter(X_sne[:, 0], X_sne[:, 1], c=y_train)
# plt.show()
# from mpl_toolkits.mplot3d import Axes3D

# 定义SNE模型


plt.rcParams["font.sans-serif"]=["Arial"]
plt.rcParams["font.family"]="Arial"
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
# 复制一份数据集，防止对原始数据进行修改

from nirs.parameters import X_train,y_train,X_test,y_test

X_train = np.concatenate((X_train,X_test),axis=0)
y_train = np.concatenate((y_train,y_test),axis=0)
x_copy = X_train.copy()
y_copy = y_train.copy()

# 定义按照y_train排序的函数
def sortByY(x, y):
    sorted_indices = np.argsort(y)
    return x[sorted_indices], y[sorted_indices]

# 按照y_train的值对x_train和y_train进行排序
x_y = np.column_stack((x_copy, y_copy))
x_y_sorted = sortByY(x_y[:, :-1], x_y[:, -1])
x_train_sorted, y_train_sorted = x_y_sorted



def draw2D(c = 2):
    sne = TSNE(n_components=2, perplexity=c, n_iter=500)

    # 训练SNE模型
    X_sne = sne.fit_transform(x_train_sorted)
    # 绘制SNE三维可视化图像
    fig = plt.figure(figsize=(9, 8))
    ax = plt.gca()
    colors = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    markers = np.array(['o', 's', '^', 'v', '*', 'x', 'D'])
    start = [0, 10, 30, 40, 50, 55, 60, 70, 100]

    a = []
    j = 0

    labels = ["Second-Step Drying", "First-Step Drying", "Rolling", "Cooling and breezing", "De-enzyming",
              "Tedding fresh leaving", "Fresh tea"]
    labels.reverse()

    x_train_reduced = X_sne
    for f, i in enumerate(y_train_sorted):
        if i >= start[j + 1]:
            j += 1
            ax.scatter(x_train_reduced[f, 0],
                       x_train_reduced[f, 1],

                       c=colors[j - 1], label=labels[j - 1], s=60 - (7 - j) * 5)
            print(labels[j - 1])
            a.append(j)
            continue
        a.append(j)
        ax.scatter(x_train_reduced[f, 0],
                   x_train_reduced[f, 1],
               s=60 - (7 - j) * 5)
    # ax.scatter(X_sne[:, 0], X_sne[:, 1], X_sne[:, 2], c=y_train)
    # ax.view_init(elev=25, azim=-45)



    font = FontProperties()
    font.set_weight('bold')
    font.set_size(12)
    font.set_family('Arial')
    ax.set_xlabel("t-SNE1", fontproperties=font)
    ax.set_ylabel('t-SNE2', fontproperties=font)

    plt.legend()
    plt.tight_layout()
    save_path = f'sne/sne_2d_visualization{c}.png'
    plt.savefig(save_path, dpi=200)
    # plt.show()

    print('可视化结果已保存在{}'.format(save_path))

    # plt.show()

for i in range(13,51,5):
    draw2D(i)
def draw3D():
    sne = TSNE(n_components=3, perplexity=25, n_iter=1000)

    # 训练SNE模型
    X_sne = sne.fit_transform(x_train_sorted)
    # 绘制SNE三维可视化图像
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111, projection='3d')
    colors =np.array( ['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    markers = np.array(['o', 's', '^', 'v', '*', 'x', 'D'])
    start = [0,10, 30, 40, 50, 55, 60, 70,100]

    a = []
    j = 0

    labels = ["Second-Step Drying", "First-Step Drying", "Rolling", "Cooling and breezing", "De-enzyming",
              "Tedding fresh leaving", "Fresh tea"]
    labels.reverse()

    x_train_reduced = X_sne
    for f,i in enumerate( y_train_sorted):
        if i >= start[j+1]:
            j+=1
            ax.scatter(x_train_reduced[f, 0],
                       x_train_reduced[f, 1],
                       x_train_reduced[f, 2],
                       c=colors[j-1], label=labels[j-1],s=60-(7-j)*5)
            print(labels[j-1])
            a.append(j)
            continue
        a.append(j)
        ax.scatter(x_train_reduced[f, 0],
                   x_train_reduced[f, 1],
                   x_train_reduced[f, 2],s=60-(7-j)*5)
    # ax.scatter(X_sne[:, 0], X_sne[:, 1], X_sne[:, 2], c=y_train)
    # ax.view_init(elev=25, azim=-45)
    ax.view_init(elev=25, azim=60)
    plt.tight_layout()

    font = FontProperties()
    font.set_weight('bold')
    font.set_size(12)
    font.set_family('Arial')
    ax.set_xlabel("t-SNE1",fontproperties=font)
    ax.set_ylabel('t-SNE2',fontproperties=font)
    ax.set_zlabel('t-SNE3'.format(a[2]),fontproperties=font)
    plt.legend()


    save_path = 'sne_3d_visualization.png'
    plt.savefig(save_path,dpi=100)
    # plt.show()

    print('可视化结果已保存在{}'.format(save_path))

    plt.show()

