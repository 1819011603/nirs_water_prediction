from scipy.spatial.distance import cdist
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


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