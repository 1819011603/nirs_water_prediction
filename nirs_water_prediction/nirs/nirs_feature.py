import inspect

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoLars

from utils import getDataIndex



#  特征对象方法 全用小写字母


class Example(BaseEstimator):
    def __int__(self,params):
        self.params = params
    def transform(self, X ,y):
        print(self.params)
        a = self.params["a"]
        print(a)
        print("转换方法")
    def fit(self, X ,y = None):
        print("fit方法")
# 自定义特征选择方法
def example(params):
    return Example(params)


def pca(params):
    from sklearn.decomposition import PCA

    return PCA(**params)
def plsr(params):
    from sklearn.cross_decomposition import PLSRegression
    return PLSRegression(**params)
def mds(params):
    from sklearn.manifold import MDS

    return MDS(**params)

def none(params):
    return None
class Lasso_(BaseEstimator):
    def __init__(self,**params):
        self.params = params
        self.index = None
    def transform(self, X ,y=None):
        if self.index is not None:
            return X[:,self.index],y
        raise "无 index"

    def fit(self, X ,y = None):
        print("fit方法")

        model = LassoLars(**self.params)
        model.fit(X, y)

        # 查看模型选择的特征系数
        coef = model.coef_

        # 查看特征系数不为零的特征索引
        feature_index = np.where(coef != 0)[0]

        print(feature_index)
        self.index = feature_index

        return X[:,feature_index],y
def lasso(params):
    return Lasso_(**params)



class FeatureSelection:
    def __init__(self, method='none',index_set = None, **kwargs):
        self.method =  method.lower()
        self.params = kwargs
        self.index = None
        if index_set == None:
            self.index_set = {"cars"}
        else:
            self.index_set = index_set
    def fit(self, X,y):
        m = self.method
        pls = None
        if m in globals().keys():
            pls = globals()[m](self.params)
        elif m in self.index_set:
            self.index = self.params["index"]

        else:
            raise "not support feature method"

        self.pls = pls
        if pls != None:
            pls.fit(X, y)



    def transform(self, X, y=None):

        item = self.method
        method = self.pls
        if item == 'none':
            pass

        elif   hasattr(method, "transform") and callable(getattr(method, "transform")):
            if len(inspect.signature(method.transform).parameters) == 2:
                X,y = method.transform(X,y)
            else:
                X = method.transform(X)
        elif  hasattr(method, "fit_transform") and callable(getattr(method, "fit_transform")):
            if len(inspect.signature(method.fit_transform).parameters) == 2:
                X, y = method.fit_transform(X, y)
            else:
                X = method.fit_transform(X)

        elif self.pls == None and self.index is not None:
            X = getDataIndex(X, self.index)
        else:
             raise "找不到"

        return X,y


