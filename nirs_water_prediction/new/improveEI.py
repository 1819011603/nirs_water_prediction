from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from nirs.parameters import  X_train,y_train,X_test,y_test

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.pipeline import Pipeline

X = X_train
y = y_train

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
import numpy as np



class Model(BaseEstimator,RegressorMixin):
    def __init__(self,model):
        self.model = model

    def predict(self,X):
        return self.model.predict(X)
    def fit(self,X,y = None):
        self.model.fit(X,y)

def predict(model):
    m = Model(model)
    from nirs.nirs_processing import dt
    from nirs.parameters import X_train,y_train,X_test,y_test
    X_train = dt(X_train)
    X_test = dt(X_test)

    m.fit(X_train,y_train)
    return m.predict(X_test)



rf = RandomForestRegressor(n_estimators=50, max_depth=9,max_features=50,random_state=42)
svr = SVR(kernel='rbf',C=973.66,gamma=3.72,epsilon=0.002)
plsr = PLSRegression(n_components=11)
bpnn = MLPRegressor(hidden_layer_sizes=(70,),learning_rate_init=0.05,activation='relu')


a = predict(rf)
b = predict(svr)

from scipy.stats import spearmanr
from scipy.stats import kendalltau

print(kendalltau(a,b))
print(spearmanr(a,b))
a = predict(plsr)
b = predict(bpnn)
print(kendalltau(a,b))


print(spearmanr(a,b))
