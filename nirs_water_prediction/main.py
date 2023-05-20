from scipy.interpolate import make_interp_spline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from nirs.nirs_feature import FeatureSelection
from nirs.nirs_models import Model
from nirs.nirs_processing import Preprocess
from utils import *
import numpy as np
from  nirs.util_paint import  *
plt.rcParams['font.size'] = 17
def paint(y_test, y_pred_test, R2, RMSECV, r2, RMSEP,RPD,all_name,val = None):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,8))
    # 构造模拟数据
    x = y_test
    y = y_pred_test

    # 增加一列常数1，以便拟合常数项b0
    X = np.c_[np.ones(x.shape[0]), x]

    # 计算回归系数
    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    beta1, beta0 = np.polyfit(x, y, 1)


    m1,m2 =np.min(x),np.max(x)

    t = (m2-m1)/15
    m1-=t
    m2+=t

    ma1,ma2 =np.min(y),np.max(y)
    u = (ma2-ma1)/15
    ma1-=u
    ma2+=u

    # mi = 0
    # max = 80

    X_ = np.linspace(m1, m2, 1000)
    Y_ = beta1*X_+beta0

    # 绘制数据散点图和回归直线


    plt.scatter(x, y,s = 50,color="none",edgecolors="black",label= "$R^2_{P}$=" + f"{r2:.4f}, RMSEP={RMSEP:.2f}%")
    if val is None:
        plt.scatter([0], [0],color="none",edgecolors="none",label="$R^2_{CV}$=" + f"{R2:.4f}, RMSECV={RMSECV:.2f}%")
    else:
        plt.scatter(val[0], val[1], color="black", edgecolors="black", label="$R^2_{CV}$=" + f"{R2:.4f}, RMSECV={RMSECV:.2}%")
    plt.scatter([0], [0],color="none",edgecolors="none",label=f"RPD={RPD:.4f}")
    if beta0 >= 0:
        plt.plot(X_, Y_, 'r', label='y=%.4fx+%.4f' % (beta1, beta0))
    else:
        plt.plot(X_, Y_, 'r', label='y=%.4fx%.4f' % (beta1, beta0))
    plt.xlim(m1,m2)
    plt.ylim(ma1,ma2)

    plt.xlabel('Measure value/%')
    plt.ylabel('Predictive value/%')

    a = "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(all_name,R2,RMSECV,r2,RMSEP,RPD)
    print(a)
    with open("a.tct","a") as f:
        f.write(a + '\n')



    plt.legend()
    eps_name = get_log_name(all_name,".eps","./eps")
    pdf_name = get_log_name(all_name,".pdf","./pdf")
    plt.savefig(pdf_name, format='pdf')
    # plt.savefig(eps_name, format='eps')

    print("pdf save in {}".format(pdf_name))
    plt.show()
    pass


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
    model = Model(method=model_str,**model_args.get(model_str,{}))

    # 获取模型的最优参数
    if para is not  None and para.optimal:



        model,best_params = model.best_para(model_str,X_train,y_train)
    if  para is not  None and  para.best_opt:
        a = -12
        b = -5
        X_train = np.concatenate((X_train,X_test[a:b]),axis=0)
        y_train = np.concatenate((y_train,y_test[a:b]),axis=0)
        a = np.arange(len(y_train))
        np.random.shuffle(a)
        X_train = X_train[a]
        y_train = y_train[a]
    start_model = time.perf_counter()

    # 交叉验证
    r2 = cross_val_score(model.model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    rmse = np.sqrt(-cross_val_score(model.model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1))


    # 预测
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred_test = model.predict(X_test)

    # 对输出进行后置预处理
    y_test = y_test
    y_pred_test = y_pred_test


    # 计算预测的R2和RMSEP
    r2_test = r2_score(y_test, y_pred_test)
    rmsep = np.sqrt(mean_squared_error(y_test, y_pred_test))


    # 运行时间
    model_time = time.perf_counter()-start_model
    total_time = time.perf_counter()-total_start




    # 评价指标
    R2 = r2.mean()
    RMSECV = rmse.mean()
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
    indicators = [ R2, RMSECV/100, r2, RMSEP/100,RPD, MAE, model_time, total_time, t_time]

    if para.__getattribute__("train_n") is not None:
        indicators.insert(0, np.sqrt(0 if r2 < 0 else r2))
        indicators.insert(0, np.sqrt(0 if R2 < 0 else R2))
        para.train_n.append(indicators)

    indicators = [f"{x:.4f}" for x in indicators]



    params_name = ["预处理参数","特征选择参数", "模型参数" ,"真实值" , "预测值"]
    params = [preprocess_args, feature_selection_args_list, ', '.join([f'{str(k)[:6]}={str(v)[:6]}' for k,v in best_params.items()]) if best_params is not None else '{}', f"[{','.join(np.array(y_test.ravel(),dtype=str))}]",f"[{','.join(np.array(y_pred_test.ravel(),dtype=str))}]"]
    params = [str(x) for x in params]


    header = ["全名","预处理","特征选择算法", "模型" , 'R2', 'RMSECV', "r2", 'RMSEP', "RPD",'MAE', "CPU时间", "流程时间", "总时间", *params_name]

    row = [all_name,"+".join(preprocess_args["method"]).upper(),"+".join(feature_selection_args["method"]).upper(),model_args["model"].upper() ,*indicators,*params]

    if  para.paint is not False:
        paint(y_test,y_pred_test,R2,RMSECV,r2,RMSEP,RPD,all_name)

    save2excel(row,header)
    print("\n\n\n")
para.train_n = []

def train_n_times(features_sel,preprocess='none', features='none', models ='plsr'):
    # preprocess = [['sg','dt']]
    # preprocess = [['sg']]
    preprocess = [[preprocess]]
    # preprocess = [['piecewise_polyfit_baseline_correction']]
    # features = [["pca"], ["cars"], ["spa"]]
    # features = [ ["pca"], ["cars","pca"]]
    # features = [["none"] , ["cars"]]

    my = features

    features = [[my]]
    from nirs.parameters import feature_selection_args
    feature_selection_args[my] = {'index': features_sel}
    feature_selection_args["index_set"].append(my)

    # features = [["new"]]
    models = [models]


    f(preprocess, features, models)

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
                    from nirs.parameters import X_train,y_train,X_test,y_test
                    main(np.array(X_train,copy=True), y_train, np.array(X_test,copy=True), y_test, preprocess_args, feature_selection_args, model_args,
                         index_set=feature_selection_args["index_set"])

                except Exception as e:
                    feature_selection_args["method"] = feature
                    print(e)
                    i+=1
                    raise str(e)
    print(f"出错次数 : {i}")
para.optimal = True
para.optimal = False
para.best_opt = True
para.best_opt = False
para.paint=False
para.paint=True
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
    # preprocess=[["MMS"],["none"],["SNV"],["MSC"] ,["SG"], ["DT"],  ["MSC","SNV"],["SG","SNV"], ["DT", "SNV"]]
    preprocess = [["none"],["msc"], ["SNV"], ["dwt"], ["d1"], ['d2'], ["sg"], ["dt"]]
    # preprocess = [['sg','dt']]
    # preprocess = [['sg']]
    # preprocess = [['dt'],['sg']]
    # preprocess = [['piecewise_polyfit_baseline_correction']]
    # preprocess = [['snv']]
    # preprocess = [['d1'],['d2']]
    # preprocess = [['dwt','sg']]
    features = [["none"],["pca"], ["cars"], ["spa"]]
    # features = [ ["pca"], ["cars","pca"]]
    # features = [["none"] , ["cars"]]
    # features = [['mp5spec_moisture']]
    features = [["none"]]
    models = ['svr']

    from urllib.request import getproxies


    # 是否需要开启参数寻优， 参数寻优的范围设置在nirs_models.py中
    para.optimal = True
    para.optimal = False
    para.best_opt = True
    para.best_opt = False
    para.paint=True
    para.paint=False






    # 批量运行程序   主方法为main方法
    f(preprocess,features,models)


    print(f"总时间: {time.time()-s:.4f}s")