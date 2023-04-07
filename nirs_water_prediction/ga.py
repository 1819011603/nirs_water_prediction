import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer

from  utils import  *
# 生成一个分类数据集
X, y = m5spec_moisture,m5spec_moisture_y
from nirs.nirs_processing import sg,dt

X = sg(dt(X))

def my_precision_score(y_true, y_pred):
    r2,rmsep,rpd= getRR_RMSE_RPD(y_true, y_pred)
    return rmsep

# 定义PLSR模型
pls = PLSRegression(n_components=2)

# 定义遗传算法参数
selector = GeneticSelectionCV(estimator=pls, cv=5, verbose=1, scoring="neg_root_mean_squared_error", max_features=50, n_population=50, crossover_proba=0.5, mutation_proba=0.03, n_generations=100, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=100, caching=True, n_jobs=1)
# selector = GeneticSelectionCV(estimator=pls, cv=10, verbose=1, scoring=make_scorer(my_precision_score), max_features=60, n_population=50, crossover_proba=0.5, mutation_proba=0.03, n_generations=100, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=None, caching=True, n_jobs=1)

# 执行特征选择
selector.fit(X, y)

# 输出选择的特征
print(selector.support_)
index = np.arange(700)[selector.support_]
print(index)
from main import feature_selection_args,f
cur = "ga"
feature_selection_args["ga"] = {"index":index}
para.optimal = True
preprocess = [["sg","dt"]]
# preprocess=[["MMS"],["none"],["SNV"],["MSC"] ,["SG"], ["DT"],  ["MSC","SNV"],["SG","SNV"], ["DT", "SNV"]]

features = [["ga"]]
# features = [["none"]]
models = ["plsr", "svr", "rbf"]
f(preprocess, features, models)
