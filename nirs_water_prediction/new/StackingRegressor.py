from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from nirs.nirs_processing import dt,sg
# 生成训练数据和标签
from nirs.parameters import *
X_train = dt(X_train)
X_test = dt(X_test)
# 定义基模型
models = [
          ('PLSR', PLSRegression(n_components=11)),
# ('poly_SVR', SVR(kernel='poly')),
    # ("lasso", Lasso()),
    # ('linear', LinearRegression()),
    #       ('SVR', SVR(kernel='rbf',C=973.66,gamma =3.72,epsilon=0.002)),
    #       ('SVR', SVR(kernel='rbf'),
          # ('BPNN', MLPRegressor(hidden_layer_sizes=(70,),learning_rate_init=0.05,activation='relu')),
          ('BPNN', MLPRegressor()),
            # ('RF', RandomForestRegressor(n_estimators=50, max_depth=9,max_features=50,random_state=42)),
    # ("AdaBoost", AdaBoostRegressor(
    #                 base_estimator=RandomForestRegressor(n_estimators=2, max_depth=11, max_features=30,
    #                                                      random_state=42), n_estimators=100, random_state=42,
    #                 learning_rate=0.1)),
           ("AdaBoost", AdaBoostRegressor(learning_rate=0.1)),
    # ("yumi_Adaboost",AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11,max_features=100), n_estimators=100, random_state=42,learning_rate=2)),
# ('yumi_SVR', SVR(kernel='rbf',C=1,epsilon=0.1)),
# ('linear_SVR', SVR(kernel='linear')),
# ('yumi_BPNN', MLPRegressor(hidden_layer_sizes=(200,),learning_rate_init=0.05,activation='relu')),
          ]

# 定义Stacking模型中的强学习器
rf = RandomForestRegressor(n_estimators=4, max_depth=7)
# rf = LinearRegression()
# rf = PLSRegression(n_components=3)
# rf = SVR(C=0.5)

# class Stacking(BaseEstimator,RegressorMixin):
#     def __init__(self, models, rf):
#         self.models = models
#         self.rf = rf
#
#     def fit(self, X, y):
#         # 利用K折交叉验证生成元数据集
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         meta_data = np.zeros((X.shape[0], len(self.models)))
#         for i, (name, model) in enumerate(self.models):
#             for train_index, test_index in kf.split(X):
#                 X_train, X_test = X[train_index], X[test_index]
#                 y_train, y_test = y[train_index], y[test_index]
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)
#                 meta_data[test_index, i] = y_pred.flatten()
#
#         # 使用元数据训练强学习器
#         self.rf.fit(meta_data, y.flatten())
#
#     def predict(self, X):
#         # 生成元数据集
#         meta_data = np.zeros((X.shape[0], len(self.models)))
#         for i, (name, model) in enumerate(self.models):
#             meta_data[:, i] = model.predict(X).flatten()
#
#         # 预测结果
#         return self.rf.predict(meta_data)

# 定义Stacking模型
class Stacking(BaseEstimator,RegressorMixin):
    def __init__(self, models, rf):
        self.models = models
        self.rf = rf

        self.name = []
        for i, (name, model) in enumerate(self.models):
            self.name.append(name)
        self.name = "_".join(self.name)
    def fit(self, X, y):
        # 利用K折交叉验证生成元数据集
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_data = np.zeros((X.shape[0], len(self.models)))
        for i, (name, model) in enumerate(self.models):
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                meta_data[test_index, i] = y_pred.flatten()

        # 使用元数据训练强学习器
        self.rf.fit(meta_data, y.flatten())

    def predict(self, X):
        # 生成元数据集
        meta_data = np.zeros((X.shape[0], len(self.models)))
        for i, (name, model) in enumerate(self.models):
            meta_data[:, i] = model.predict(X).flatten()

        # 预测结果
        return self.rf.predict(meta_data)
result = [models]
# result = []
# A = models
# for i in range(len(A)):
#     for j in range(i+1, len(A)):
#         for k in range(j+1, len(A)):
#             # for m in range(k+1,len(A)):
#             #     sublist = [A[i], A[j], A[k],A[m]]
#                 sublist = [A[i], A[j], A[k]]
#                 result.append(sublist)

cv = KFold(n_splits=4, shuffle=True, random_state=3)
for r in result:
    # 训练Stacking模型并预测结果
    stacking = Stacking(r, rf)

    r2 = cross_val_score(stacking ,X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    r2 = np.mean(r2)
    rmse = np.sqrt(-cross_val_score(stacking, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1))
    rmse = np.mean(rmse)
    stacking = Stacking(r, rf)
    stacking.fit(X_train, y_train)



    y_pred_test = stacking.predict(X_test)

    r2_test = r2_score(y_test, y_pred_test)
    rmsep = np.sqrt(mean_squared_error(y_test, y_pred_test))
    RPD = np.std(y_test) / rmsep
    from main import paint
    # paint(y_test, y_pred_test, r2, rmse / 100, r2_test, rmsep / 100, RPD, stacking.name,[y_train,stacking.predict(X_train)])
    paint(y_test, y_pred_test, r2, rmse / 100, r2_test, rmsep / 100, RPD, stacking.name,None)