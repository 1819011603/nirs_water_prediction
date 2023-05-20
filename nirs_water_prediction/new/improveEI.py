import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeRegressor

plt.rcParams["font.family"] = "SimHei"
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False

# plt.rcParams['font.size'] = 20


import numpy as np
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LassoLars, Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.decomposition import PCA
from nirs.nirs_processing import *


class Model(BaseEstimator,RegressorMixin):
    def __init__(self,model):
        self.model = model

    def predict(self,X):
        return self.model.predict(X)
    def fit(self,X,y = None):
        self.model.fit(X,y)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
def predict(model):
    m = Model(model)
    from nirs.nirs_processing import dt
    from nirs.parameters import X_train,y_train,X_test,y_test
    X_train1 = dt(X_train)
    X_test1 = dt(X_test)
    # p = PCA(n_components=50)
    # X_train1 = p.fit_transform(X_train1)
    # X_test1 = p.transform(X_test1)

    m.fit(X_train1,y_train)
    return m.predict(X_test1)
    # return cross_val_predict( m,X_train, y_train, cv=cv, n_jobs=-1)

# 模型列表
model_list = [
    ("AGA-PLSR", PLSRegression(n_components=11)),
    ("AGA-SVR",SVR(kernel='rbf',C=973.66,gamma=3.72,epsilon=0.002)),
    ("AGA-BPNN",MLPRegressor(hidden_layer_sizes=(70,), learning_rate_init=0.05, activation='relu',random_state=42)),
    ("AGA-RF",RandomForestRegressor(n_estimators=50, max_depth=9,max_features=50,random_state=42)),
    # ("Lasso",Lasso(alpha=0.1)),
    ("AGA-AdaBoost",AdaBoostRegressor(base_estimator= RandomForestRegressor(n_estimators=2, max_depth=11, max_features=30, random_state=42), n_estimators=100, random_state=42, learning_rate=0.1)),


              ]


# model_list = [   ('PLSR', PLSRegression(n_components=15)),
#                  ('linear', LinearRegression()),
#                  # ("lasso",Lasso(alpha=0.1)),
#           # ('SVR', SVR(kernel='rbf',C=973.66,gamma =3.72,epsilon=0.002)),
#           # ('BPNN', MLPRegressor(hidden_layer_sizes=(70,),learning_rate_init=0.05,activation='relu')),
#             ('RF', RandomForestRegressor(n_estimators=50, max_depth=9,max_features=50,random_state=42)),
#     # ("AdaBoost", AdaBoostRegressor(
#     #                 base_estimator=RandomForestRegressor(n_estimators=2, max_depth=11, max_features=30,
#     #                                                      random_state=42), n_estimators=100, random_state=42,
#     #                 learning_rate=0.1)),
#     # ("yumi_Adaboost",AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11,max_features=100), n_estimators=100, random_state=42,learning_rate=2)),
# ('yumi_SVR', SVR(kernel='rbf',C=1,epsilon=0.1)),
# ('poly_SVR', SVR(kernel='poly')),
# ('linear_SVR', SVR(kernel='linear')),
# # ('yumi_BPNN', MLPRegressor(hidden_layer_sizes=(200,),learning_rate_init=0.05,activation='relu')),
# ]
# 创建Adaboost模型



model_label=["PLSR", "SVR", "BPNN","RF","Lasso","Adaboost"]

# 初始化相关系数矩阵
n = len(model_list)
kendall_matrix = np.zeros((n, n))

# 计算Kendall相关系数矩阵

dic = {}
for i in range(n):
    for j in range(n):
        x =dic.get(i,predict(model_list[i][1]))
        y =dic.get(j,predict(model_list[j][1]))
        corr, p_value = kendalltau(x, y)
        kendall_matrix[i, j] = corr
        if dic.get(i) is None:
            dic.setdefault(i,x)
        if dic.get(j) is None:
            dic.setdefault(j, y)

# print(kendall_matrix)


corr = kendall_matrix

# 绘制相关系数热力图
fig, ax = plt.subplots()
# im = ax.imshow(corr, cmap='seismic', vmin=0.4, vmax=1) # cmap设置为冷暖色调，vmin和vmax设置颜色映射的范围
im = ax.imshow(corr, cmap='coolwarm', vmin=0.4, vmax=1) # cmap设置为冷暖色调，vmin和vmax设置颜色映射的范围
# im = ax.imshow(corr, cmap='RdBu_r', vmin=0.4, vmax=1) # cmap设置为冷暖色调，vmin和vmax设置颜色映射的范围
# im = ax.imshow(corr, cmap='Spectral_r', vmin=0.4, vmax=1) # cmap设置为冷暖色调，vmin和vmax设置颜色映射的范围
# im = ax.imshow(corr, cmap='PuOr_r', vmin=0.4, vmax=1) # cmap设置为冷暖色调，vmin和vmax设置颜色映射的范围

# 添加颜色柱子
cbar = plt.colorbar(im)
# cbar.set_label('Kendall correlation coefficient')
cbar.set_label('Kendall相关系数')

# 设置坐标轴标签和标题
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels([f'{model_list[i][0]}' for i in range(n)])
ax.set_yticklabels([f'{model_list[i][0]}' for i in range(n)])
# ax.set_title('Spearman correlation between {} models'.format(n))

# 在热力图上添加文本标签
for i in range(n):
    for j in range(n):
        text = ax.text(j, i, '{:.2f}'.format(corr[i, j]), ha='center', va='center', color='black')

plt.savefig("热力图.pdf")
plt.show()
