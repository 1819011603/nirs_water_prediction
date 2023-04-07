import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score


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
        "C": trial.suggest_loguniform('C', 1e-3, 2e3),
        "gamma": trial.suggest_loguniform('gamma', 1e-3, 2e3),
        "epsilon": trial.suggest_loguniform('epsilon', 1e-3, 1e0),
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
def bpnn_set(trial,**param):
    return {
                "hidden_layer_sizes" :( trial.suggest_int("hidden_layer_sizes", 8,40) ,),

                "activation" :trial.suggest_categorical("activation", [ 'logistic', 'tanh', 'relu']),
                "solver" :trial.suggest_categorical("solver", ['sgd', 'adam']),
                "learning_rate_init" :  trial.suggest_loguniform('learning_rate_init', 1e-3, 1),
                 "max_iter" : trial.suggest_categorical("max_iter",[5000,10000]),
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
            alpha = self.params['alpha']
            self.model = Lasso(alpha=alpha)
        elif self.method == 'svr':
            from sklearn.svm import SVR

            self.model = SVR(**self.params)
        elif self.method == 'bpnn':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                    **self.params

)
        elif self.method == "plsr":
            from sklearn.cross_decomposition import PLSRegression

            self.model = PLSRegression(**self.params)
        elif self.method == 'rbf':
            self.model = RBFNet(**self.params)

        else:
            raise ValueError('Unsupported model method.')

    #     预测的时候才要用
    def fit(self, X, y):
        self.model.fit(X, y)

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
            scores = cross_val_score(estimator.model, X_, Y_, cv=5, scoring='neg_mean_squared_error')

            return -scores.mean()

            # Create an instance of the Optuna study object

        study = optuna.create_study(direction='minimize')

        # Run the optimization process for 100 trials
        study.optimize(objective, n_trials=50,n_jobs=1)
        # study.optimize(objective, n_trials=min(150,len(X_[0])-9),n_jobs=1)

        # Print the best hyperparameters found by Optuna
        print("Best hyperparameters: ", study.best_params)
        return  Model(method=model_str,**study.best_params), study.best_params

