from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class ELMRegressor(BaseEstimator,RegressorMixin):
    def __init__(self, n_hidden, C=1):
        self.n_hidden = n_hidden
        self.C = C
        self.beta = None

    def relu(self, X):
        return np.maximum(X, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.random_weights = np.random.normal(size=[n_features, self.n_hidden])
        H = self.relu(X.dot(self.random_weights))
        self.beta = np.linalg.inv(H.T.dot(H) + np.eye(self.n_hidden) / self.C).dot(H.T).dot(y)

    def predict(self, X):
        H = self.relu(X.dot(self.random_weights))
        y_pred = H.dot(self.beta)
        return y_pred.flatten()
class BaggingRegressor(BaseEstimator,RegressorMixin):
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

import numpy as np

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Stacking(BaseEstimator,RegressorMixin):
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

class BPNN(nn.Module,RegressorMixin,BaseEstimator):
    def __init__(self, hidden_size,learning_rate=0.01,num_epochs=10000):
        super(BPNN, self).__init__()

        self.hidden_size = hidden_size
        self. num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.hidden_layer = None
        self.output_layer = None
        self.activation=None
    def forward(self, x):

        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def fit(self, X, y):
        _, input_size = X.shape
        self.hidden_layer = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self.activation = nn.Tanh()

        X = torch.tensor(X,dtype=torch.float)
        y = torch.tensor(y,dtype=torch.float)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,betas=(0.999,0.937))
        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for i in range(self.num_epochs):
            y_predict = self.forward(X)
            loss = criterion(y_predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            outputs = self.forward(X)
            return outputs.numpy()


class DeepELM(RegressorMixin,BaseEstimator):
    def __init__(self, hidden_sizes, alpha=0.1, regularization=None, lambda_val=None):

        self.hidden_sizes = hidden_sizes

        self.alpha = alpha
        self.regularization = regularization
        self.lambda_val = lambda_val
        self.weights = []
        self.biases = []
        self.beta = None
        self.sample = None

    def init_weights(self):
        self.weights.append(np.random.normal(size=(self.input_size, self.hidden_sizes[0])))
        self.biases.append(np.random.normal(size=self.hidden_sizes[0]))
        for i in range(1, len(self.hidden_sizes)):
            self.weights.append(np.random.normal(size=(self.hidden_sizes[i-1], self.hidden_sizes[i])))
            self.biases.append(np.random.normal(size=self.hidden_sizes[i]))
        self.weights.append(np.random.normal(size=(self.hidden_sizes[-1], self.output_size)))
        self.biases.append(np.random.normal(size=self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y):

        self.sample= X.shape[0]

        self.input_size=X.shape[1]
        self.output_size=1
        self.init_weights()

        self.hidden_layers = []
        X_train = X.copy()
        for i in range(len(self.weights)):
            layer_input = np.dot(X_train, self.weights[i]) + self.biases[i]
            X_train = self.relu(layer_input)
            self.hidden_layers.append(X_train)
        if self.regularization is not None:
            if self.regularization == 'l1':
                reg = np.sum([np.sum(i) for i in np.abs(self.weights)])
            elif self.regularization == 'l2':
                reg = np.sum([np.sum(i) for i in np.square(self.weights)])
            else:
                raise ValueError('Invalid regularization parameter')
            reg *= self.lambda_val
        else:
            reg = 0
        H = np.hstack(self.hidden_layers)
        self.beta = np.linalg.pinv(np.dot(H.T, H) + reg)
        self.beta = np.dot(self.beta, np.dot(H.T, y))

    def smote_upsample(self,X, y, target_samples):
        """使用 SMOTE 算法进行上采样，并将样本数量增加到目标数"""
        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        n_samples = X_resampled.shape[0]

        if n_samples > target_samples:
            # 如果生成的样本数量已经超过了目标样本数，随机选择部分样本
            idx = np.random.choice(n_samples, size=target_samples, replace=False)
            X_resampled = X_resampled[idx, :]
            y_resampled = y_resampled[idx]

        return X_resampled, y_resampled
    def predict(self, X,y = None):
        if self.sample != None:
            # X,y=self.smote_upsample(X,y,self.sample)
            self.sample = None
        X_test = X.copy()
        for i in range(len(self.weights)):
            layer_input = np.dot(X_test, self.weights[i]) + self.biases[i]
            X_test = self.relu(layer_input)
        H = np.hstack(self.hidden_layers)
        y_pred = np.dot(H, self.beta)


        return y_pred.ravel()

class R_ELM(BaseEstimator,RegressorMixin):
    def __init__(self,  n_hidden, reg=1e-6):

        self.n_hidden = n_hidden

        self.reg = reg

        self.bias = np.random.randn(n_hidden)
        self.beta = None

    def relu(self, x):
        return np.maximum(x, 0)

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.n_input = n_features
        self.n_output=1
        self.weights = np.random.randn(self.n_input, self.n_hidden)
        H = self.relu(np.dot(X, self.weights) + self.bias)
        H_t = np.transpose(H)
        self.beta = np.dot(np.linalg.inv(np.dot(H_t, H) + self.reg * np.identity(self.n_hidden)), np.dot(H_t, Y))

    def predict(self, X):
        H = self.relu(np.dot(X, self.weights) + self.bias)
        Y_pred = np.dot(H, self.beta)
        return Y_pred


class LSSVM(BaseEstimator,RegressorMixin):
    def __init__(self, ** params):
        self.kernel = params.get('kernel','rbf')
        self.sigma = params.get('sigma',1)
        self.gamma = params.get('gamma',1)
        self.C = params.get('C',1)
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        K = self.kernel_matrix(X_train)
        self.alpha = np.linalg.solve(K + self.C * np.identity(K.shape[0]), y_train)
        self.b = np.mean(y_train - np.dot(K, self.alpha))

    def predict(self, X_test):
        K_test = self.kernel_matrix(X_test, self.X_train)
        return np.dot(K_test, self.alpha) + self.b

    def kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if self.kernel == 'rbf':
            return np.exp(-cdist(X1, X2) ** 2 / (2 * self.sigma ** 2))
        elif self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X1, X2.T) + 1) ** self.sigma

if __name__ == '__main__':
    from nirs.parameters import *
    from nirs.nirs_processing import sg,dt
    bpnn=BPNN(50,0.001)
    X_train = dt(sg(X_train))
    y_train = y_train/100
    bpnn.fit(X_train,y_train)
    print(bpnn.predict(X_train))

