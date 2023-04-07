
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from nirs.nirs_feature import FeatureSelection
from nirs.nirs_models import Model
from nirs.nirs_processing import Preprocess
from utils import *
import numpy as np

def main(X_train, y_train, X_test, y_test, preprocess_args, feature_selection_args, model_args,index_set=None):
    import time
    t_start = time.time()
    total_start = time.perf_counter()

    # preprocess
    preprocessor = Preprocess(**preprocess_args)
    X_train = preprocessor.transform(X_train,y_train)
    X_test = preprocessor.transform(X_test,y_test)

    # feature selection
    # 获取当前的特征选择方法列表
    index = feature_selection_args["method"]
    feature_selection_args.pop("method")
    feature_selection_args_list = []
    for m in index:

        feature_selection_args_list.append(feature_selection_args.get(m))
        feature_selector = FeatureSelection(method=m,index_set=index_set,**feature_selection_args.get(m,{}))
        feature_selector.fit(X_train,y_train)
        X_train, y_train = feature_selector.transform(X_train, y_train)
        X_test, y_test = feature_selector.transform(X_test, y_test)

    feature_selection_args["method"] = index

    # cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=3)

    model_str = model_args["model"].lower()

    best_params = model_args.get(model_str)
    # 模型
    model = Model(method=model_str,**model_args.get(model_str))

    # 获取模型的最优参数
    if para.optimal:
        model,best_params = model.best_para(model_str,X_train,y_train)

    start_model = time.perf_counter()

    # 交叉验证
    r2 = cross_val_score(model.model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    rmse = np.sqrt(-cross_val_score(model.model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1))


    # 预测
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred_test = model.predict(X_test)

    # 对输出进行后置预处理
    y_test = y_test / 100
    y_pred_test = y_pred_test / 100


    # 计算预测的R2和RMSEP
    r2_test = r2_score(y_test, y_pred_test)
    rmsep = np.sqrt(mean_squared_error(y_test, y_pred_test))


    # 运行时间
    model_time = time.perf_counter()-start_model
    total_time = time.perf_counter()-total_start




    # 评价指标
    R2 = r2.mean()
    RMSECV = rmse.mean()/100
    r2 = r2_test
    RMSEP= rmsep
    RPD = np.std(y_test)/RMSEP

    MAE = mean_absolute_error(y_test, y_pred_test)
    #     将模型的参数保存至
    t_time = time.time() - t_start

    all_name = ("_".join([*preprocess_args["method"], *feature_selection_args["method"], model_args["model"]])).upper()
    print(f"model: {all_name}")
    print(f'R2: {R2:.4f}')
    print(f'RMSECV: {RMSECV:.4f} ')
    print(f"r2: {r2:.4f}")
    print(f"RMSEP: {RMSEP:.4f}")
    print(f"RPD: {RPD:.4f}")
    print(f"MAE: {MAE:.4f}")
    print(f"model_time: {model_time:.4f}s")
    print(f"total_time: {total_time:.4f}s")
    print(f"t_time: {t_time:.4f}s")

    # 指标列表
    indicators = [ R2, RMSECV, r2, RMSEP, MAE, model_time, total_time, t_time]
    indicators = [f"{x:.4f}" for x in indicators]

    params_name = ["预处理参数","特征选择参数", "模型参数" ,"真实值" , "预测值"]
    params = [preprocess_args, feature_selection_args_list, best_params, y_test, y_pred_test]
    params = [str(x) for x in params]


    header = ["全名","预处理","特征选择算法", "模型" , 'R2', 'RMSECV', "r2", 'RMSEP', 'MAE', "CPU时间", "流程时间", "总时间", *params_name]

    row = [all_name,"+".join(preprocess_args["method"]).upper(),"+".join(feature_selection_args["method"]).upper(),model_args["model"].upper() ,*indicators,*params]
    save2excel(row,header)


    print("\n\n\n")


def f(preprocess,features,models):


    i= 0


    for model in models:

        model_args["model"] = model
        from nirs.parameters import para
        # 结果的保存路径， 在result文件夹下
        if not hasattr(para, "optimal"):
            para.optimal = False
        para.path = f"tmp_{model}{'_opt' if para.optimal else ''}.xlsx"
        for feature in features:
            feature_selection_args["method"] = feature
            for p in preprocess:
                preprocess_args["method"] = p
                try:
                    main(X_train, y_train, X_test, y_test, preprocess_args, feature_selection_args, model_args,
                         index_set=feature_selection_args["index_set"])

                except Exception as e:
                    feature_selection_args["method"] = feature
                    print(e)
                    i+=1
                    raise str(e)
    print(f"出错次数 : {i}")

if __name__ == '__main__':






    # SG + DT + PCA + SVR

    # preprocess=[ ["SG","DT"] ]
    # features = [["pca"]]
    # models = ["svr"]

    #
    import time
    s = time.time()


    # 模型参数修改在parameters.py中
    # preprocess=[ ["baseline_correction"]]
    preprocess=[["MMS"],["none"],["SNV"],["MSC"] ,["SG"], ["DT"],  ["MSC","SNV"],["SG","SNV"], ["DT", "SNV"]]
    features = [["none"],["cars"], ["pca"], ["cars","pca"]]
    features = [ ["pca"], ["cars","pca"]]
    features = [  ["none"]]
    # features = [["none"]]
    models = ["plsr"]


    # 是否需要开启参数寻优， 参数寻优的范围设置在nirs_models.py中
    para.optimal = True
    # para.optimal = False



    # 批量运行程序   主方法为main方法
    f(preprocess,features,models)


    print(f"总时间: {time.time()-s:.4f}s")