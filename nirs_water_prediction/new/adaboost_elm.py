import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from nirs.my_model import ELMRegressor
# from nirs.nirs_feature import pca
from nirs.parameters import X_train,y_train,X_test,y_test
# 构建ELM模型
# elm1 = ELMRegressor(n_hidden=100)
# elm2 = ELMRegressor(n_hidden=200)
# elm3 = ELMRegressor(n_hidden=1000)
#
# # 构建AdaBoostClassifier模型
# boosted_elm = AdaBoostRegressor(base_estimator=elm3, n_estimators=1000)
# # boosted_elm = GradientBoostingRegressor(base_estimator=elm3, n_estimators=1000)
#
# # 训练模型
# boosted_elm.fit(X_train, y_train)
#
# # 预测
# y_pred = boosted_elm.predict(X_test)
#
#
# print(r2_score(y_test,y_pred))
# print(y_pred)



# 构建AdaBoostClassifier模型
# boosted_elm = AdaBoostRegressor(n_estimators=1000,base_estimator=)
# boosted_elm = GradientBoostingRegressor(base_estimator=elm3, n_estimators=1000)
lasso = Lasso(alpha=0.1)

# 使用决策树回归作为基学习器
from nirs.nirs_processing import dt
X_train = dt(X_train)
X_test = dt(X_test)
dt = DecisionTreeRegressor(max_depth=10)

# 使用AdaBoostRegressor集成多个基学习器
boosted_elm = AdaBoostRegressor(base_estimator=dt, n_estimators=100, learning_rate=0.05, loss='linear')
boosted_elm = AdaBoostRegressor(base_estimator= RandomForestRegressor(n_estimators=2, max_depth=11, max_features=30, random_state=42), n_estimators=100, random_state=42, learning_rate=0.1)
# 训练模型
boosted_elm.fit(X_train, y_train)
# print(boosted_elm.feature_importances_)


def top_n_indices(arr, n):
    """
    输出一个numpy数组前n大数的索引

    参数：
    arr -- 一个numpy数组
    n -- 前n大数的数量

    返回值：
    一个列表，包含前n大数在原数组中的索引
    """
    indices = np.argsort(arr)[::-1][:n]
    return indices.tolist()

importance = boosted_elm.feature_importances_
indice = np.array(top_n_indices(importance,50))
indice.sort()
from nirs.parameters import xpoints
print(xpoints[indice])
print(np.cumsum(importance[indice]))
# 预测
y_pred = boosted_elm.predict(X_test)


print(r2_score(y_test,y_pred))
# print(y_pred)
y_pred = boosted_elm.predict(X_train)

print(r2_score(y_train,y_pred))
# print(y_pred)