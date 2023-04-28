from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from nirs.nirs_processing import dt
# 生成训练数据和标签
from nirs.parameters import *
X_train = dt(X_train)
X_test = dt(X_test)
# 定义基模型
models = [
          # ('plsr', PLSRegression(n_components=11)),
          ('svr', SVR(kernel='rbf',C=973.66,gamma=3.72,epsilon=0.002)),
          ('bpnn', MLPRegressor(hidden_layer_sizes=(70,),learning_rate_init=0.05,activation='relu')),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=9,max_features=50,random_state=42))
          ]

# 定义Stacking模型中的强学习器
rf = RandomForestRegressor(n_estimators=4, max_depth=7)
# rf = LinearRegression()
# rf = PLSRegression(n_components=3)
# rf = SVR(kernel="poly",C=100)


# 定义Stacking模型
class Stacking:
    def __init__(self, models, rf):
        self.models = models
        self.rf = rf

    def fit(self, X, y):
        # 利用K折交叉验证生成元数据集
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
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


# 训练Stacking模型并预测结果
stacking = Stacking(models, rf)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_train)

# 计算均方误差
mse =np.sqrt( mean_squared_error(y_train, y_pred))
print("Mean squared error: %.4f" % mse)


y_pred = stacking.predict(X_test)

print(r2_score(y_test, y_pred))
# 计算均方误差
mse =np.sqrt( mean_squared_error(y_test, y_pred))
print("Mean squared error: %.4f" % mse)
