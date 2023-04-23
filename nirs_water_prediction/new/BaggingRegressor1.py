import numpy as np
from sklearn.tree import DecisionTreeRegressor

class BaggingRegressor:
    def __init__(self, n_estimators=10, max_samples=1.0, max_features=1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.estimators_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        for i in range(self.n_estimators):
            # 训练基模型
            estimator = DecisionTreeRegressor(max_features=30,max_depth=15)
            # 随机采样样本和特征
            sample_indices = np.random.choice(n_samples, size=int(self.max_samples * n_samples), replace=False)
            # feature_indices = np.random.choice(n_features, size=int(self.max_features * n_features), replace=False)
            # X_subset = X[sample_indices][:, feature_indices]
            X_subset = X[sample_indices]
            y_subset = y[sample_indices]
            # 训练基模型
            estimator.fit(X_subset, y_subset)
            # 将基模型加入集成中
            self.estimators_.append(estimator)

    def predict(self, X):
        # 预测结果为基模型的平均值
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        for i, estimator in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X)
        return np.mean(predictions, axis=1)


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from nirs.parameters import *

    from nirs.nirs_processing import *

    X_train = dt(X_train)
    X_test = dt(X_test)

    # 生成一组样本数据
    # X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=1)

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # 创建Bagging模型
    model = BaggingRegressor(n_estimators=60, max_samples=0.8, max_features=0.8)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    print(r2_score(y_test,y_pred))
    # 计算均方误差
    rmse =np.sqrt( mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
