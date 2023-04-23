from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.linear_model import LinearRegression
import numpy as np
from nirs.parameters import  X_train,y_train,X_test,y_test

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.pipeline import Pipeline

X = X_train
y = y_train

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
import numpy as np



# 创建随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 创建逐步特征选择对象
rfecv = RFECV(estimator=rf, step=4, cv=5, scoring='r2')

# 拟合逐步特征选择对象
rfecv.fit(X, y)

# 输出选择的特征的索引
print("最优特征数目: %d" % rfecv.n_features_)
print("最优特征排名: %s" % rfecv.ranking_)
print(np.arange(256)[rfecv.support_])
