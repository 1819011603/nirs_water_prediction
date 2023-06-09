import numpy as np
import sklearn.linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from nirs.RegressionNet import Regression
from nirs.my_model import  *
from nirs.util_paint import *
from utils import get_log_name


class RBFNet(BaseEstimator,RegressorMixin):
    def __init__(self, hidden_shape, sigma=1.0,alpha=0.1,kernel="gaussian",gamma=1):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.alpha=alpha
        self.kernel=kernel
        self.gamma=gamma
        self.centers = None
        self.weights = None

    def _kernel_function(self, X1, X2):
        if self.kernel == "gaussian":
            from sklearn.metrics.pairwise import rbf_kernel

            return np.exp(-self.sigma*np.linalg.norm(X1-X2)**2)
        elif self.kernel == 'poly':
            degree = 30
            coef0 = 1
            gamma = 1
            return (gamma*np.dot(X1, X2.T) + coef0) ** degree



    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        return G

    def fit(self, X, Y):
        # kmeans = KMeans(n_clusters=self.hidden_shape, init=bik, random_state=0).fit(X)
        kmeans = KMeans(n_clusters=self.hidden_shape, init='k-means++', random_state=0).fit(X)

        # kmeans = KMeans(n_clusters=self.hidden_shape, init='random', random_state=0).fit(X)

        self.centers = kmeans.cluster_centers_

        # 获取聚类中心点

        G = self._calculate_interpolation_matrix(X)

        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        return np.dot(G, self.weights)
def svr_set(trial,**param):
    return {
        "C": trial.suggest_loguniform('C', 1e-7, 7e7),
        "gamma": trial.suggest_loguniform('gamma', 1e-7, 1e4),
        "epsilon": trial.suggest_loguniform('epsilon', 1e-7, 1e2),
        "kernel": trial.suggest_categorical("kernel", ['rbf']),

    }


def lssvm_set(trial,**param):
    return {
        "C": trial.suggest_loguniform('C', 1e-3, 2e3),
        "gamma": trial.suggest_loguniform('gamma', 1e-3, 2e3),
        "sigma": trial.suggest_loguniform('epsilon', 1e-3, 1e0),
        "kernel": trial.suggest_categorical("kernel", ['rbf']),

    }
def plsr_set(trial,**param):

    return  {
                "n_components": trial.suggest_int("n_components", 8,  param.get("len")),
            }
def rbf_set(trial,**param):
    return {
                "hidden_shape": trial.suggest_int("hidden_shape", 8, 256),
                'sigma': trial.suggest_loguniform('sigma', 1e-3, 2e3),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 2e3),
                "kernel": trial.suggest_categorical("kernel", ['gaussian', "poly"]),
                "gamma": trial.suggest_loguniform('gamma', 1e-3, 2e3),
                # 'hidden_layer_sizes': [(10,), (50,), (100,)],
            }

def rf_set(trial,**param):
    return {
        'n_estimators' : trial.suggest_int("n_estimators", 30, 150, step=10),
   'max_depth' : trial.suggest_int("max_depth", 3, 7),
    'min_samples_split' : trial.suggest_int("min_samples_split", 2, 5),
    'min_samples_leaf' : trial.suggest_int("min_samples_leaf", 1, 5),
    'max_features' : trial.suggest_categorical("max_features", ["sqrt", "log2"])
            }
def bpnn_set(trial,**param):
    return {
                "hidden_size" :( trial.suggest_int("hidden_size", 20,100) ,),

                # "activation" :trial.suggest_categorical("activation", [ 'logistic', 'tanh', 'relu']),
                # "solver" :trial.suggest_categorical("solver", ['sgd', 'adam']),
                "learning_rate" :  trial.suggest_loguniform('learning_rate', 1e-3, 0.2),
                 "num_epochs" : trial.suggest_categorical("num_epochs",[5000,10000]),
            }


# 参数范围映射， 新增的参数搜索需要在这里定义关系
function_sets = {"svr": svr_set, "plsr":plsr_set, "bpnn":bpnn_set, "rbf": rbf_set}




class Model:
    def __init__(self, method= None, **kwargs):
        self.method = method
        self.params = kwargs
        if self.method == 'linear':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        elif self.method == 'ridge':
            from sklearn.linear_model import Ridge
            alpha = self.params['alpha']
            self.model = Ridge(alpha=alpha)
        elif self.method == 'lasso':
            from sklearn.linear_model import Lasso
            alpha = self.params.get('alpha',0.1)
            self.model = Lasso(alpha=alpha)
        elif self.method == 'svr':
            from sklearn.svm import SVR

            self.model = SVR(**self.params)
        elif self.method == 'bpnn':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=(30,),learning_rate_init=0.01,activation='relu',max_iter=1000,  n_iter_no_change=800)
            self.model = MLPRegressor(hidden_layer_sizes=(20,),learning_rate_init=0.1,activation='relu',max_iter=1000,  n_iter_no_change=800)
            self.model = MLPRegressor()
            # self.model = BPNN(**self.params)
        elif self.method == "plsr":
            from sklearn.cross_decomposition import PLSRegression

            self.model = PLSRegression(**self.params)
        elif self.method == 'rbf':
            self.model = RBFNet(**self.params)

        elif self.method=='lar':
            from sklearn.linear_model import LassoLars

            # 构造LassoLars对象
            self.model = LassoLars(**self.params)
        elif self.method == 'lssvm':
            from nirs.my_model import LSSVM
            self.model = LSSVM(**self.params)
        elif self.method == 'rf':
             self.model =  RandomForestRegressor(random_state=42)

        elif self.method == 'bagging':
             self.model =   BaggingRegressor(n_estimators=60, max_samples=0.8, max_features=0.8)
        elif self.method == 'xgb':

             import xgboost as xgb
             self.model = xgb_reg = xgb.XGBRegressor(
                 objective='reg:squarederror',  # 损失函数为平方误差
                 learning_rate=0.15,  # 学习率
                 max_depth=2,  # 决策树最大深度
                 n_estimators=400,  # 弱学习器个数

                 subsample=1,
                 colsample_bytree=0.6

                 # early_stopping_rounds=1000,
             )
        elif self.method == 'stacking':
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.neural_network import MLPRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import Lasso
            models = [
                # ('plsr', PLSRegression(n_components=11)),
                ('svr', SVR(kernel='rbf', C=973.66, gamma=3.72, epsilon=0.002)),
                # # ('bpnn', MLPRegressor(hidden_layer_sizes=(100,50)))
                # ('rf', RandomForestRegressor(n_estimators=50, max_depth=9, max_features=50, random_state=42)),

                # ("PLSR", PLSRegression(n_components=11)),
                # ("AGA-SVR", SVR(kernel='rbf', C=973.66, gamma=3.72, epsilon=0.002)),
                # ("AGA-BPNN", MLPRegressor(hidden_layer_sizes=(70,), learning_rate_init=0.05, activation='relu',random_state=42)),
                # ("RF", RandomForestRegressor(n_estimators=50, max_depth=9, max_features=50, random_state=42)),
                # ("Lasso",Lasso(alpha=0.1)),
                ("AdaBoost", AdaBoostRegressor(
                    base_estimator=RandomForestRegressor(n_estimators=2, max_depth=11, max_features=30,
                                                         random_state=42), n_estimators=100, random_state=42,
                    learning_rate=0.1)),

            ]

            # 定义Stacking模型中的强学习器
            rf = RandomForestRegressor(n_estimators=4, max_depth=7)
            self.model =   Stacking(models, rf)
        elif self.method == 'gbdt':
             self.model =  GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, max_depth=4, max_features=30,
                                       subsample=1.0, random_state=42, criterion='squared_error', min_samples_split=2)
        elif self.method == 'adaboost':


            base_estimator = RandomForestRegressor(n_estimators=2, max_depth=11, max_features=30, random_state=42)

            # 创建Adaboost模型



            # self.model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
            # base_model = sklearn.linea
            # 构建AdaBoost回归模型
            self.model =   AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, random_state=42, learning_rate=0.1)
            self.model = AdaBoostRegressor(random_state=42)
            # self.model= GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif self.method == 'ielm':
            # self.model = ELMRegressor(100)
            self.model = R_ELM(100)
            # self.model = DeepELM((70,32),regularization='l1',lambda_val=2)





        elif self.method == 'xgboost':
            import xgboost
            self.model = xgboost.XGBRegressor(
                n_estimators=1000,  # 基学习器数量
                learning_rate=0.1,  # 学习率
                max_depth=7,  # 最大树深
                min_child_weight=3,  # 叶子节点最小权重和
                subsample=0.8,  # 随机采样比例
                colsample_bytree=0.8,  # 每棵树随机采样比例
                objective='reg:squarederror',  # 目标函数为均方误差
                random_state=42  # 随机数种子
            )

        else:
            raise ValueError('Unsupported model method.')

    #     预测的时候才要用
    def fit(self, X, y):

        self.model.fit(X, y)

        if self.method == 'bpnn1':

            curse = np.array(self.model.loss_curve_)/10000
            all_name = "myarray"

            pdf_name = get_log_name(all_name,".txt","./pdf")
            np.savetxt(pdf_name, curse,fmt='%.6f')

            fig = plt.figure(figsize=(8, 8), dpi=150)
            plt.plot(range( self.model.max_iter), curse)
            plt.xlabel('Iteration')
            plt.ylabel('Training Error')
            plt.show()

            plt.xlabel('Measure value/%')
            plt.ylabel('Predictive value/%')

            plt.legend()
            all_name = "bpnn"
            eps_name = get_log_name(all_name,".eps","./eps")
            pdf_name = get_log_name(all_name,".pdf","./pdf")
            plt.savefig(pdf_name, format='pdf')
            # plt.savefig(eps_name, format='eps')

            print("pdf save in {}".format(pdf_name))
            plt.show()
    def predict(self, X):

        return self.model.predict(X)

    def best_para(self,model_str,X_,Y_):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)


        import optuna
        def objective(trial):
            # Define the hyperparameters to be optimized by Optuna
            # Define the hyperparameters to be optimized by Optuna

            # 设置待调参数的范围
            param_grid= None
            if model_str+"_set" in globals().keys():
                param_grid = globals()[ model_str+"_set"](trial,len=len(X_[0]))
            else:
                raise "unsupported model  for search parameters, please complete the model_str+ '_set' method"

            estimator =  Model(method=model_str,**param_grid)
            # Evaluate the estimator using cross-validation
            # a = np.arange(len(Y_))
            # np.random.shuffle(a)
            # X_ = X_[a,:]
            # Y_ = Y_[a,:]
            scores = cross_val_score(estimator.model, X_, Y_, cv=5, scoring='neg_mean_squared_error')
            return -scores.mean()


            # i = 400
            # estimator.fit(X_[:i],Y_[:i])
            # j = 400
            # y_pred = estimator.predict(X_[j:-80])
            #
            # return mean_squared_error(Y_[j:-80], y_pred)

            # Create an instance of the Optuna study object

        study = optuna.create_study(direction='minimize')

        # Run the optimization process for 100 trials
        study.optimize(objective, n_trials=30,n_jobs=1)
        # study.optimize(objective, n_trials=min(150,len(X_[0])-9),n_jobs=1)

        # Print the best hyperparameters found by Optuna
        print("Best hyperparameters: ", study.best_params)
        return  Model(method=model_str,**study.best_params), study.best_params

