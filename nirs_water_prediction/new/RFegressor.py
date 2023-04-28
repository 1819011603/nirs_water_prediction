from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nirs.parameters import *
from nirs.nirs_processing import *
X_train = dt(X_train)
X_test = dt(X_test)
from sklearn.svm import SVR
base_estimator = DecisionTreeRegressor(max_depth=11,max_features=50)
# 创建SVR基学习器
# base_estimator = SVR(kernel='rbf',C=50,gamma=5)
base_estimator = SVR(kernel='rbf',C=50,gamma=15)
ada = RandomForestRegressor(n_estimators=50, max_depth=9,max_features=50,random_state=42)



# 训练模型
ada.fit(X_train, y_train)

# 测试模型
y_pred = ada.predict(X_test)
y_train_pred = ada.predict(X_train)
print(r2_score(y_train, y_train_pred))

print(r2_score(y_test, y_pred))
# 计算均方误差
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
