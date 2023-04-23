from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from nirs.my_model import ELMRegressor
from nirs.nirs_feature import pca
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
dt = DecisionTreeRegressor(max_depth=10)
X_train = PCA(X_train)
X_test = pca(X_test)

# 使用AdaBoostRegressor集成多个基学习器
boosted_elm = AdaBoostRegressor(base_estimator=dt, n_estimators=100, learning_rate=0.05, loss='linear')

# 训练模型
boosted_elm.fit(X_train, y_train)

# 预测
y_pred = boosted_elm.predict(X_test)


print(r2_score(y_test,y_pred))
print(y_pred)
y_pred = boosted_elm.predict(X_train)

print(r2_score(y_train,y_pred))
print(y_pred)