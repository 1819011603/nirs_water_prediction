import pathlib

import numpy
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from numpy import *
import scipy
from scipy import signal
import pandas as pd
# 数据读取-单因变量与多因变量
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# 根据分隔符读取数据
from sklearn.metrics import mean_squared_error


def loadDataSet01(filename, Separator=', '):
    fr = open(filename)
    arrayLines = fr.readlines()
    assert len(arrayLines) != 0
    num = len(arrayLines[0].split(Separator)) - 1
    row = len(arrayLines)
    x = mat(zeros((row, num)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split(Separator)
        curLine = [float(i) for i in curLine]
        x[index, :] = curLine[0: -1]
        # y[index, :] = curLine[-1]/100
        y[index, :] = curLine[-1]
        index += 1

    return np.array(x), np.array(y)

# 求x,y的 均值与方差
def data_Mean_Std(x0, y0):
    mean_x = mean(x0, 0)
    mean_y = mean(y0, 0)
    std_x = std(x0, axis=0, ddof=1)
    std_y = std(y0, axis=0, ddof=1)
    return mean_x, mean_y, std_x, std_y

import pywt

def DWT(x0):
    wavename = 'db5'
    cA, cD = pywt.dwt(x0, wavename)
    x0 = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
    # x0 = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
    return x0

# 数据标准化
from sklearn import preprocessing, metrics


def stardantDataSet(x0, y0):
    e0 = preprocessing.scale(x0)
    f0 = preprocessing.scale(y0)
    return e0, f0

# 中值滤波
def med_filtering(tea):
    ans = signal.medfilt(tea, 3)
    return ans

# 高斯滤波
def gaussian_filtering(tea):
    ans = scipy.ndimage.filters.gaussian_filter(tea, sigma=0.85, mode="nearest")
    return ans

# 信号的最小二乘平滑 是一种在时域内基于局域多项式最小二乘法拟合的滤波方法。这种滤波器最大的特点在于在滤除噪声的同时可以确保信号的形状、宽度不变。
def savitzky_golay(x0):  # 实现曲线平滑
    for x in range(len(x0)):
        x0[x] = signal.savgol_filter(x0[x], 7, 3, mode="wrap")
    return x0

def detrend(x0):
    for x in range(len(x0)):
        x0[x] = signal.detrend(x0[x])
    return x0

def MSC(data_x):  # 多元散射校正
    ## 计算平均光谱做为标准光谱
    mean = numpy.mean(data_x, axis=0)

    n, p = data_x.shape
    msc_x = numpy.ones((n, p))

    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_  # 线性回归系数
        b = lin.intercept_ # 线性回归截距
        msc_x[i, :] = (y - b) / k
    return msc_x

def EMSC(data_x):  # 多元散射校正
    ## 计算平均光谱做为标准光谱
    mean = numpy.mean(data_x, axis=0)

    n, p = data_x.shape
    msc_x = numpy.ones((n, p))

    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_  # 线性回归系数
        b = lin.intercept_ # 线性回归截距
        msc_x[i, :] = (y - b) / k
    return msc_x


# 标准正态变换
def SNV(data):
    m = data.shape[0]
    n = data.shape[1]
    print(m, n)  #
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return np.array( data_snv)


# MSC(数据)
def MSC(Data):
    # 计算平均光谱
    n, p = Data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(Data, axis=0)

    # 线性拟合
    for i in range(n):
        y = Data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return np.array(msc)

# 一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return np.array(Di)

# 二阶导数
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di
# 移动平均平滑
def MA(a, WSZ=5):
    for i in range(a.shape[0]):
        out0 = np.convolve(a[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(a[i, :-WSZ:-1])[::2] / r)[::-1]
        a[i] = np.concatenate((start, out0, stop))
    return np.array(a)

# 最大最小值归一化
def MMS(data):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    return MinMaxScaler().fit_transform(data)

# 信号预处理
def filter(x0):

    # if True:
    #     return  x0

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    xx = np.zeros(shape=(x0.shape[0],x0.shape[1]-2))
    # print(xx.shape)
    # x0 = D1(x0)
    for x in range(len(x0)):
        x0[x] = savitzky_golay(x0[x])
        # x0[x] = gaussian_filtering(x0[x])
        x0[x] = detrend(x0[x])

        # x0[x] = med_filtering(x0[x])  # 并没有提升
        xx[x] = x0[x][1:-1]

    # x0 = preprocessing.scale(x0) # 标准化
    # x0 = SNV(x0)
    # return xx
    # x0 = MMS(x0)
    return x0

def filter(x0,filter_method=None):
    if filter_method is None:
        filter_method=[savitzky_golay, detrend]
        # filter_method = [DWT]
    # if True:
    #     return  x0

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # print(xx.shape)
    # x0 = D1(x0)

    for method in filter_method:

        x0 = method(x0)



    # x0 = preprocessing.scale(x0) # 标准化
    # x0 = SNV(x0)
    # return xx
    # x0 = MMS(x0)
    return x0

import torch
# 获得RR RMSE
RPD_total = 0
def getRR_RMSE(y_test,y_predict,isVal_ = False):
    global RPD_total
    if isinstance(y_test, torch.Tensor):
        row = len(y_test)
        y_mean = torch.mean(y_test, 0).clone().detach()
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        RMSE = sqrt(SSE / row)
        if isVal_:
            # RPD_total += 1 / sqrt(1 - RR)
            RPD_total += torch.std(y_predict)/ RMSE
        return RR, RMSE
        # return sqrt(RR), RMSE

    y_test = y_test.ravel()
    y_predict = y_predict.ravel()
    y_mean = mean(y_test, 0)
    row = len(y_test)
    SSE = sum(sum(power((y_test - y_predict), 2), 0))
    SST = sum(sum(power((y_test - y_mean), 2), 0))
    # SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    # print(SSE, SST, SSR)
    from sklearn.metrics import r2_score
    RR = 1 - (SSE / SST)
    # RR = r2_score(y_test.ravel(),y_predict.ravel())
    # print(RR == r2_score(y_test.ravel(),y_predict.ravel()) )
    RMSE = sqrt(SSE / row)
    # print((RMSE - np.sqrt(mean_squared_error(y_test, y_predict)) == 0 ))
    # RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
    if isVal_:
        # RPD_total += 1/sqrt(1-RR)
        # print( np.std(y_predict), RMSE,  np.std(y_predict)/RMSE  )
        RPD_total += np.std(y_predict)/RMSE
        # print(np.std(y_predict)/RMSE)
    # print("RPD:",1/sqrt(1-RR))RR
    return RR,RMSE
    # return sqrt(RR), RMSE


def drawPCA2d(x_train,  y_train,color_shuffle=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    #  PCA 聚类
    colors = ["b","g","r","c","m","y","k","brown","greenyellow",'lightgreen','midnightblue','tomato','skyblue']
    # colors = ["b","g","r","c","m","y","k",'midnightblue','tomato']
    markers = ["o", "v","^","s",">","D"]
    fig=plt.figure(figsize=(12, 9), dpi=600)
    from mpl_toolkits.mplot3d import Axes3D
    #  三维
    # ax = Axes3D(fig)

    # 二维
    ax = plt.gca()



    y_s = []
    t = []
    j = 0
    for i in colors:

        t.append( (i,markers[j% len(markers)]))
        j+=1
    #
    if color_shuffle:
        random.shuffle(t)



    len1 = 10

    dic ={}
    idx = 0
    for i,x_ in enumerate(x_train[:]):
        y_= int(y_train[i][0])


        #  三维
        if int(y_/len1) not in y_s:
            dic[int(y_/len1) * len1] = t[idx]
            idx+=1

            ax.scatter(x_[0], x_[1],  c= dic[int(y_/len1) * len1 ][0], marker= dic[int(y_/len1) * len1][1], label=str(int(y_/len1) * len1) + "~" + str(int(y_/len1) * len1 + len1 - 1) + "%",
                          )
        else:
            ax.scatter(x_[0], x_[1], c= dic[int(y_/len1) * len1 ][0], marker= dic[int(y_/len1) * len1][1])


        # 二维
        # if y_ not in y_s:
        #
        #     ax.scatter(x_[0], x_[1], c=t[y_ % len(t)][0], label=str(y_), marker=t[y_ % len(t)][1])
        # else:
        #
        #     ax.scatter(x_[0], x_[1], c=t[y_ % len(t)][0], marker=t[y_ % len(t)][1])
        y_s.append(int(y_/10) )
    ax.patch.set_facecolor("white")
    # 三维
    # ax.set_zlabel('PC3(2.73%)', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('PC2(20.82%)', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('PC1(74.82%)', fontdict={'size': 15, 'color': 'black'})
    #  标签生效
    plt.legend()
    plt.savefig(get_log_name("PCA","png","./PCA"))

    plt.show()



def drawPCA3d(x_train,  y_train,color_shuffle=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    #  PCA 聚类
    colors = ["b","g","r","c","m","y","k","brown","greenyellow",'lightgreen','midnightblue','tomato','skyblue']
    # colors = ["b","g","r","c","m","y","k",'midnightblue','tomato']
    markers = ["o", "v","^","s","h"]
    fig=plt.figure(figsize=(12, 9), dpi=600)
    from mpl_toolkits.mplot3d import Axes3D
    #  三维
    ax = Axes3D(fig)

    # 二维
    # ax = plt.gca()



    y_s = []
    t = []
    for i in markers:
        for j in colors:
            t.append((j,i))
    if color_shuffle:
        random.shuffle(t)



    len1 = 10

    dic ={}


    idx = 0
    for i,x_ in enumerate(x_train[:]):
        y_= int(y_train[i][0])


        #  三维
        if int(y_/len1) not in y_s:
            dic[int(y_/len1) * len1] = t[idx]
            idx+=1

            ax.scatter(x_[0], x_[1],x_[2],   c= dic[int(y_/len1) * len1 ][0], marker= dic[int(y_/len1) * len1][1], label=str(int(y_/len1) * len1) + "~" + str(int(y_/len1) * len1 + len1 - 1)+ "%",
                        )
        else:
            ax.scatter(x_[0], x_[1],x_[2],  c= dic[int(y_/len1) * len1 ][0], marker= dic[int(y_/len1) * len1][1])


        # 二维
        # if y_ not in y_s:
        #
        #     ax.scatter(x_[0], x_[1], c=t[y_ % len(t)][0], label=str(y_), marker=t[y_ % len(t)][1])
        # else:
        #
        #     ax.scatter(x_[0], x_[1], c=t[y_ % len(t)][0], marker=t[y_ % len(t)][1])
        y_s.append(int(y_/10) )
    ax.patch.set_facecolor("white")
    # 三维
    ax.set_zlabel('PC3(2.73%)', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('PC2(20.82%)', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('PC1(74.82%)', fontdict={'size': 15, 'color': 'black'})
    #  标签生效
    plt.legend()
    plt.savefig(get_log_name("PCA","png","./PCA"))

    plt.show()

def PCA(x_train, x_test, y_train, y_test,x_val,y_val,n_components=100):
    from sklearn.decomposition import PCA
    # pls2 = PCA(copy=True,n_components=11, tol=1e-06)
    pls2 = PCA(copy=True,n_components=n_components)
    pls2.fit(x_train)

    # 贡献率
    # print(np.round(pls2.explained_variance_ratio_,4))
    # print(np.round(np.cumsum(pls2.explained_variance_ratio_),4))





    # #  相关性矩阵
    # relative = np.around(np.corrcoef(x_train.T), decimals=3)

    # plt.figure(figsize=(10,10),dpi=600)
    # sns.set(color_codes=True)
    # sns.heatmap(relative, square=True, annot=True)
    # plt.title("Correlation matrix")
    # plt.show()

    x_train = pls2.transform(x_train)


    #  画2维图像
    # drawPCA2d(x_train, y_train)
    #  画3维图像
    # drawPCA3d(x_train, y_train)

    x_test = pls2.transform(x_test)
    x_val = pls2.transform(x_val)




    # print(pls2.components_)
    # print(pls2.explained_variance_)
    from biPls import regressionNet,LS_SVM
    return x_train,x_test,y_train,y_test,x_val,y_val #[41,109]  99 100 159
    # return PLS(x_train,x_test,y_train,y_test,x_val,y_val)  # 102
    # return Linear_Regression(x_train,x_test,y_train,y_test,x_val,y_val) # 102

def SPA(x_train, x_test, y_train, y_test,x_val,y_val):
    import PLS.SPA


    # y_predict = pls2.transform(x_test)
    # print(y_predict.shape)
    # print(pls2.components_)
    # print("PCA:" , pls2.explained_variance_ratio_,len(pls2.explained_variance_ratio_))
    # print(pls2.explained_variance_)

    # y_val_ = pls2.predict(x_val)
    # RR,RMSE = getRR_RMSE(y_test,y_predict)
    # RR1,RMSE1 = getRR_RMSE(y_val,y_val_)

    # return RR, RMSE,RR1,RMSE1


def write_to_csv(y_predict,y_test,file_path="./PLS-master/data/Sigmoid_Sigmoid_99.36.pkl"):
    y_predict = torch.reshape(torch.Tensor(y_predict),(-1,1))
    y_test = torch.Tensor(y_test)
    y_predict = torch.concat(( y_test,y_predict, y_predict - y_test), dim=1)
    y_predict = pd.DataFrame(y_predict.detach().numpy())
    ff = get_log_name(pre="net", suff="csv", dir_path=str(pathlib.Path(file_path).parent))
    print("csv file save in {}".format(ff))
    y_predict.to_csv(ff)
import logging
def log_mod(name):
    import logging
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
    # logger = logging.getLogger(name)
    # logger.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # fh = logging.FileHandler('access.log',encoding='utf-8')
    # fh.setLevel(logging.WARNING)
    # ch_formatter = logging.Formatter('%(module)s-%(lineno)d %(levelname)s:%(message)s')
    # fh_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s:%(message)s',datefmt='%Y/%m/%d %H:%M:%S')
    # ch.setFormatter(ch_formatter)
    # fh.setFormatter(fh_formatter)
    # logger.addHandler(ch)
    # logger.addHandler(fh)
    # # 这里需要把logger返回
    # return logger


def paint(y_test,y_predict,y_val,y_val_, name,RP2,csv_p=None,filter_method=None):
    import tropic

    if csv_p is None:
        csv_p="PLS/PLS-master/data"
    # 测试集
    csv_path = save_to_csv(csv_p, y_true=y_val, y_predict=y_val_,pre="predict_",suff="_"+name+".csv",filter_method=filter_method)

    # 校正集
    csv_path1 = save_to_csv(csv_p, y_true=y_test, y_predict=y_predict,pre="val_",suff="_"+name+".csv",filter_method=filter_method)
    # tropic.show_tropic(csv_path,csv_path1=csv_path1,pictrue_name="PLS/PLS-master/data/picture",name=name,RP2=RP2)


def biPLS_LS_SVM(x_train, x_test, y_train, y_test,x_val,y_val,name="biPLS+SVR",csv_p=None,filter_method=None):

    return LS_SVM(x_train, x_test, y_train, y_test, x_val, y_val, name, csv_p, filter_method);

def LS_SVM(x_train, x_test, y_train, y_test,x_val,y_val,name="SVR",csv_p=None,filter_method=None):
    from sklearn import svm
    # pls2 = svm.LinearSVR(C=700,tol=1e-6)
    # pls2 = svm.SVR(kernel='rbf',C=1e2,gamma=48)
    # pls2 = svm.SVR(kernel='rbf',C=1000,gamma=48)


    # [698.31031586  88.89026228]   0.9899
    # [1670.17818722   93.76367181]  0.9898

    # savitzky_golay,detrend
    pls2 = svm.SVR(kernel='rbf',C=1250,gamma=48,epsilon=0.1)
    pls2 = svm.SVR(kernel='rbf',C=990,gamma=92.4,epsilon=0.1)
    # pls2 = svm.SVR(kernel='rbf')

    # pls2 = svm.SVR(kernel='rbf',C=26,gamma=1.6)
    # pls2 = svm.SVR(kernel='linear',C=1000)

    # {'C': 99, 'gamma': 17, 'kernel': 'rbf'}   DWT
    # pls2= svm.SVR(kernel="rbf", C=99,gamma=17,epsilon=0.1)



    # pls2 = svm.SVR(kernel='rbf',C=5000,gamma=48,epsilon=0.0014)
    # {'C': 950, 'gamma': 38, 'kernel': 'rbf'}
    # pls2 = svm.SVR(kernel='rbf',C=950,gamma=38)
    # pls2 = svm.SVR(kernel='linear',C=1019,gamma=20)
    # pls2 = svm.SVR(kernel='rbf',C=10000,gamma=48)
    # pls2 = svm.SVR(kernel='rbf',C=1e5,gamma=2.4)
    # pls2 = svm.SVR(kernel='poly',C=20000000000,degree=2)
    pls2.fit(x_train, y_train.ravel())
    y_predict = pls2.predict(x_test)
    y_train_ = pls2.predict(x_train)
    # print(pls2.coef_)
    # print(pls2.intercept_)

    # print("y_predict:" ,y_predict)
    # print("y_test:" , y_test.ravel())
    y_val_ = pls2.predict(x_val)


    import tropic
    RR, RMSE = getRR_RMSE(y_test.ravel(), y_predict)
    RR1, RMSE1 = getRR_RMSE(y_val.ravel(), y_val_,True)

    # 保存csv
    # tropic.show_tropic(csv_path,RC2=RR,RP2=RR1,RMSEC=RMSE,RMSEP=RMSE1,pictrue_name="PLS/PLS-master/data/picture")
    # 保存csv

    # 保存结果
    # paint(y_test,y_predict,y_val,y_val_,name=name, RP2=RR1,csv_p=csv_p,filter_method=filter_method)


    # paint(y_train,y_train_,y_val,y_val_,"SVR")
    # csv_path = save_to_csv("PLS/PLS-master/data", y_true=y_val, y_predict=y_val_)
    # tropic.show_tropic(csv_path,pictrue_name="PLS/PLS-master/data/picture",name="SVR")

    # write_to_csv(y_val_,y_val)
    # write_to_csv(y_predict,y_test)
    return RR, RMSE, RR1, RMSE1


#  https://blog.csdn.net/qq_41815357/article/details/109637463
def randomForest(x_train, x_test, y_train, y_test, x_val, y_val,name="RF",csv_p=None,filter_method=None):
    # forest = RandomForestRegressor() # 95.58%
    # forest = RandomForestRegressor(n_estimators=60,oob_score=True, criterion="mse")  # 96.08%
    # forest = RandomForestRegressor(n_estimators=30,oob_score=True, criterion="mse") #r2 96.19%
    # forest = RandomForestRegressor(n_estimators=min(15,len(x_train[0])), criterion="mse",min_samples_split=2,min_samples_leaf=2,max_features=min(11,len(x_train[0])),bootstrap=False)
    forest = RandomForestRegressor(n_estimators=min(38,len(x_train[0])), criterion="mse",min_samples_split=2,min_samples_leaf=2, max_depth=12 ,max_features=min(5,len(x_train[0])),bootstrap=False)
    forest.fit(x_train, y_train.ravel())

    y_predict = forest.predict(x_test)

    # print("y_predict:" ,y_predict)
    # print("y_test:" , y_test.ravel())
    y_val_ = forest.predict(x_val)

    RR, RMSE = getRR_RMSE(y_test.ravel(), y_predict)
    RR1, RMSE1 = getRR_RMSE(y_val.ravel(), y_val_, True)
    paint(y_test, y_predict.ravel(), y_val, y_val_.ravel(), name, RP2=RR1,csv_p=csv_p,filter_method=filter_method)
    # write_to_csv(y_val_,y_val)
    # write_to_csv(y_predict,y_test)
    return RR, RMSE, RR1, RMSE1

def PCA_randomForest(x_train, x_test, y_train, y_test, x_val, y_val,n_components=31,name="PCA_randomForest"):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,n_components=n_components)
    return randomForest(x_train, x_test, y_train, y_test, x_val, y_val,name=name)

def PCA_LS_SVM(x_train, x_test, y_train, y_test,x_val,y_val,n_components=31,name="PCA+SVR"):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,n_components=n_components)
    return LS_SVM(x_train, x_test, y_train, y_test, x_val, y_val,name)
def PCA_PLS(x_train, x_test, y_train, y_test,x_val,y_val,n_components=31,name="PCA_PLS"):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,n_components=n_components)
    return PLS(x_train, x_test, y_train, y_test, x_val, y_val,name=name)

def PCA_ELM(x_train, x_test, y_train, y_test,x_val,y_val,n_components=31,name="PCA_ELM"):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,n_components=n_components)
    return ELM(x_train, x_test, y_train, y_test, x_val, y_val,n_components= n_components,name=name)
def PCA_BP_NNN(x_train, x_test, y_train, y_test,x_val,y_val,n_components=31,name="PCA+BP_DNN"):
    x_train, x_test, y_train, y_test, x_val, y_val = PCA(x_train, x_test, y_train, y_test, x_val, y_val,
                                                         n_components=n_components)
    from biPls import regressionNet
    return regressionNet(x_train, x_test, y_train, y_test, x_val, y_val,name)
def Linear_Regression(x_train, x_test, y_train, y_test,x_val,y_val):


    pls2 = LinearRegression()

    pls2.fit(x_train, y_train)
    # print(pls2.coef_)
    # print(pls2.intercept_)
    # print(pls2.score(x_train,y_train))
    y_predict = pls2.predict(x_test)
    y_val_ = pls2.predict(x_val)
    RR, RMSE = getRR_RMSE(y_test, y_predict)
    RR1, RMSE1 = getRR_RMSE(y_val, y_val_,True)
    return RR, RMSE, RR1, RMSE1
def save_to_csv(csv_path, y_true, y_predict,pre="net",suff="csv",filter_method=None):
    y_true = torch.tensor(y_true)
    # 增加维度 (92,) -> (92,1)
    y_predict = torch.tensor(y_predict).unsqueeze(1).expand(-1,1)
    f= pathlib.Path(csv_path)
    ans = pd.DataFrame(torch.concat((y_true,y_predict , y_predict - y_true), dim=1).detach().numpy())
    ff = get_log_name(pre=pre, suff=suff, dir_path=str(f),filter_method=filter_method)
    print("csv file save in {}".format(ff))
    ff = str(pathlib.Path(ff).absolute())
    ans.to_csv(ff)
    return ff

def toXlsx(x0, force = False,filter_method=None,dir_path="./filter"):
    #  force 为True 表示强制重写
    import openpyxl,os
    cur_dir= os.path.split(os.path.realpath(__file__))[0]
    xlsx_path = get_log_name(pre="filter",suff="_biPLS.xlsx",dir_path=dir_path, filter_method=filter_method)
    if force is False and os.path.exists(cur_dir + os.path.sep + xlsx_path):
        return
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    for i,row in enumerate(x0):
        for j,column in enumerate(row):
            worksheet.cell(i+1, j + 1).value = column
    workbook.save(xlsx_path)
def PLS1(x_train, x_test, y_train, y_test):
    RR = 0
    RMSE = 0

    start = min(8, len(x_train[0]))
    num = 1
    for i in range(start, start + num):
        pls2 = PLSRegression(n_components=i, max_iter=750, tol=1e-06, scale=True)
        pls2.fit(x_train, y_train)
        y_predict = pls2.predict(x_test)
        RR,RMSE = getRR_RMSE(y_test,y_predict)
    return RR, RMSE


def ELM(x_train, x_test, y_train, y_test, x_val, y_val,n_components=256, name="ELM",csv_p=None,filter_method=None):
    import hpelm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    import time
    Train_T = []
    Test_E = []
    Test_R = []

    ##Load wine testing UCI data data
    # data = np.genfromtxt('winequality-white.csv', dtype=float, delimiter=';')
    #
    # # Delete heading
    # data = np.delete(data, 0, 0)
    #
    # x = data[:, :11]
    # y = data[:, -1]
    #
    # # Train test split
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # ===============================================================

    def calculateE(y, t):
        # Calculate RMSE
        return mean_squared_error(y, t)

    def calculateR(y_test, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(y_test, y_pred)

    # ===============================================================
    # Initialization

    # 隐藏层  最大神经元个数
    rand_num = len(x_train[0])
    Lmax = int(rand_num/1.2)
    # Lmax = 200
    L = 0
    E = 0
    R = 0
    ExpectedAccuracy = 0

    # 核函数  神经元类型：“lin”表示线性，“sigm”或“tanh”表示非线性，“rbf_l1”、“rbf_l2”或“rbf_linf”表示径向基函数神经元。
    # kernel_function="sigm"
    # kernel_function="rbf_l1"
    # kernel_function="rbf_l2"
    # kernel_function="rbf_linf"
    kernel_function="tanh"
    # kernel_function="lin"

    while L < Lmax and E >= ExpectedAccuracy:
        # Increase Node
        L = L + 1

        # Calculate Random weights, they are  already addded into model using hpelm library
        w = random.rand(rand_num, L)

        # Initialize model

        model = hpelm.ELM(rand_num, 1)
        model.add_neurons(L,kernel_function)

        start_time = time.time()

        # Train model
        model.train(x_train, y_train, 'r')

        Train_T.append(time.time() - start_time)

        # Calculate output weights and intermediate layer
        BL_HL = model.predict(x_test)

        # Calculate new EMSE
        R,E = getRR_RMSE(y_test, BL_HL, False)
        # E = calculateE(y_test, BL_HL)
        # R = calculateR(y_test, BL_HL)
        Test_E.append(E)
        Test_R.append(R)

        # Print result
        # print("Hidden Node", L, "RMSE :", E, "R2: ", R)

    # ===================================================================
    L = Test_E.index(min(Test_E)) + 1

    print()
    print()
    print()
    print()

    # Define model
    model = hpelm.ELM(rand_num, 1)
    model.add_neurons(L, kernel_function)

    start_time = time.time()
    model.train(x_train, y_train, 'r')
    print('Training Time :', time.time() - start_time)

    start_time = time.time()
    BL_HL = model.predict(x_train)
    print('Testing Time :', time.time() - start_time)
    y_predict = model.predict(x_test)
    # Calculate training RMSE
    E = calculateE(y_train, BL_HL)
    print('Training RMSE :', E)
    RMSECV = min(Test_E)
    RC2 = Test_R[Test_E.index(RMSECV)]
    print('Testing RMSE  :  {}, R2:{}'.format(RMSECV, RC2))

    y_val_ = model.predict(x_val)

    RP2,RMSEP =  getRR_RMSE(y_val, y_val_,True)
    # paint(y_test,  y_predict.ravel(), y_val, y_val_.ravel(), name, RP2=RP2,csv_p=csv_p,filter_method=filter_method)
    return RC2, RMSECV, RP2, RMSEP


def PLS(x_train, x_test, y_train, y_test,x_val,y_val, n_components=8,name="PLS",csv_p=None,filter_method=None):
    RR = 0
    RMSE = 0
    RR1 = 0
    RMSE1 = 0

    PC1 = min(len(x_train[0]), n_components)

    num = 1
    for i in range( PC1,  PC1+ num):
        pls2 = PLSRegression(n_components=i, max_iter=750, tol=1e-06, scale=True)
        pls2.fit(x_train, y_train)


        y_predict = pls2.predict(x_test)
        y_val_ = pls2.predict(x_val)
        RR, RMSE = getRR_RMSE(y_test, y_predict)
        RR1, RMSE1 = getRR_RMSE(y_val, y_val_,True)

        # paint(y_test, y_predict.ravel(), y_val, y_val_.ravel(), name, RP2=RR1,csv_p=csv_p,filter_method=filter_method)
    return RR, RMSE, RR1, RMSE1




def split10items(x0, y0, splitss=10, random_state=1, extend=1):
    import random
    # random.seed(random_state)
    len = np.shape(x0)[0]
    a = list(np.arange(0, len))
    # random.shuffle(a)
    u = 0
    r = int(len / splitss)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(splitss - 1):
        x_test.append(x0[a[u:u + r]])
        y_test.append(y0[a[u:u + r]])

        train_x =a[0:u]
        train_x.extend( list(a[u + r:]))

        if extend > 1:
            p = list(train_x)
            for i in range(1, extend):
                train_x.extend(p)
        x_train.append(x0[train_x])
        y_train.append(y0[train_x])
        u += r

    x_test.append(x0[a[u:]])
    y_test.append(y0[a[u:]])
    x_train.append(x0[a[0:u]])
    y_train.append(y0[a[0:u]])
    return x_train, x_test, y_train, y_test


def Cross_validation(x0, y0, f_test, splits=10, random_state=11, extend=1):
    x0 = filter(x0)
    x_trains, x_tests, y_trains, y_tests = split10items(x0, y0, splits=splits, random_state=random_state, extend=extend)
    p = 0
    m = 0
    for i in range(len(x_trains)):
        a, b = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
        p += a
        m += b

    print(u"R^2 {0}%".format(np.round(p / len(x_trains) * 100, 2)))
    print(u"RMSE.", m / len(x_trains))
j = 0

def get_log_name(pre="recode",suff= "log",dir_path="./",filter_method=None):
    import pathlib
    import re
    kk = re.compile("(\d+)")
    o=[]
    # print(suff)
    if suff.rfind(".") >=0 :
        o = [str(i.stem) for i in pathlib.Path(dir_path).glob("{}*{}".format(pre,suff[suff.rfind(".")+1:]))]
    else:
        o = [str(i.stem) for i in pathlib.Path(dir_path).glob("{}*{}".format(pre, "." + suff))]
    import datetime
    day = str(datetime.date.today())
    day = day[day.index('-')+1:].replace("-","_")
    max1 = 0
    for po in o:
        u = re.search(kk, po)
        if u != None:
            m = int(u.group(0))
            max1 = max(m, max1)
    f = pathlib.Path(dir_path)
    if not f.exists():
        f.mkdir(parents=True)
    if suff.rfind(".") < 0:

        return "{}/{}{}_{}.{}".format(f,pre,max1 + 1,day,suff)
    else:
        # return "{}/{}{}_{}{}".format(f,pre,max1 + 1,day,suff)
        filter_methods=[]
        if filter_method is None:
            filter_method=[savitzky_golay, detrend]
        for i,method in enumerate(filter_method):
            filter_methods.append(filter_method[i].__name__)
        return "{}/{}{}_{}_{}{}".format(f,pre,max1 + 1,"_".join(filter_methods),day,suff)
def getSplitsAndIndices(split_len=10):
    l = np.shape(x0)[1]
    len1 = int(np.ceil(l / split_len))  # 剩余的尾巴不要了
    a = list(np.arange(l))
    b = list(np.arange(len1))
    splits = []
    u = 0
    for i in range(len1):
        splits.append(a[u:u + split_len])
        u += split_len
    splits.append(a[u:])
    return splits, b
from sklearn.model_selection import train_test_split
def main1(s_len = 11):
    import time
    start = time.time()
    global x0,bb,splits

    x0, y0 = loadDataSet01('./PLS-master/data/train.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0,None)


    splits, bb = getSplitsAndIndices(split_len=s_len)

    # print(getNext(1))
    # print(getNext(2))
    # print(getNext(3))
    # print(getNext(23))
    m = 0
    m_j = 0
    b_ = []

    rm = 10
    rm_j = 1
    mylog = open(get_log_name(), mode='a', encoding='utf-8')

    while len(bb) > ceil(8.0 / s_len):
        k = 0
        p = get_iter(x0)
        max = 0
        max_j = 0
        for x in p:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y0, test_size=0.2, random_state=3)

            RR, RMSE = PLS1(x_train, x_test, y_train, y_test)
            print("{} RR: {} RMSE: {}".format(k, RR, RMSE), file=mylog)
            if max < RR:
                max = RR
                max_j = k
            if rm > RMSE:
                rm = RMSE
                rm_j = len(bb)
            k += 1

        if m < max or abs(max-m) < 0.001:

            m_j = len(bb)
            b_ = list(bb)
            if m < max:
                m = max
        print("max_RR: {}, delete group is {}".format(max,bb[max_j]), file=mylog)
        bb.remove(bb[max_j])
        print(bb, file=mylog)

    print(file=mylog)
    print('the best groups: {}'.format(b_), file=mylog)
    print("R2_max:{}, b_len: {}".format(m, m_j), file=mylog)
    print("rmse_min: {}, b_len: {}".format(rm,rm_j),file=mylog)
    # print(bb,file=mylog)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)),file=mylog)
    mylog.close()
    # print(x)
    # print(np.shape(x))
    # break
    # print(len)
    # print(x0)


#  每组11个
def main2(index = list(range(0,256)),s_len = 11):
    import time
    start = time.time()
    index = np.array(index)
    global x0,bb,splits

    x0, y0 = loadDataSet01('./PLS-master/data/train.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)

    x0 = getDataIndex(x0,index)

    splits, bb = getSplitsAndIndices(split_len=s_len)

    # print(getNext(1))
    # print(getNext(2))
    # print(getNext(3))
    # print(getNext(23))
    m = 1e10
    m_j = 0
    b_ = []

    bb_len = len(bb)

    mylog = open(get_log_name(), mode='a', encoding='utf-8')

    biPls_log = open("biPls.txt", mode='a', encoding='utf-8')
    print("\n\n\n\n", file=biPls_log)
    order = 1
    order_ = 1
    while len(bb) > 1:

        # k表示删除哪个区间的RMSE最小
        k = 0
        p = get_iter(x0)
        max = 0
        max_j = 0
        rm = 10
        rm_j = 1

        idx1=[]

        for idx,x in p:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y0, test_size=0.3,random_state=3)
            # x1, y1 = loadDataSet01("./PLS-master/data/test.txt", ", ")
            # x1 = filter(x1, filter_method=None)
            # mm1 = getDataIndex(x1,index[ idx])
            RR, RMSE = PLS1(x_train, x_test, y_train, y_test)
            # print("{} RR: {} RMSE: {}".format(k, RR, RMSE), file=mylog)
            if max < RR:
                max = RR
                max_j = k
            if rm > RMSE:
                rm = RMSE
                rm_j = k
                idx1=idx
            k += 1

        print("Min_RMSE: {}, delete group is {}, cur group is {}".format(rm,bb[rm_j],index[idx1]), file=mylog)


        print("{} {} {} {}".format(order,bb[rm_j],rm, len(idx1)),file=biPls_log)

        bb.remove(bb[rm_j])
        # print(bb, file=mylog)


        if m > rm :
            m_j =  len(bb)
            b_ = idx1
            m = rm
            order_=order
        order += 1
    print("\n{}".format(order_), file=biPls_log)
    print(file=mylog)
    print('the best groups: {}'.format(index[b_]), file=mylog)
    print("find Min_RMSE: {}, b_len: {}".format(m, m_j), file=mylog)
    from biPls import cross
    cross(PLS,3,index1=index[b_].tolist())
    cross(LS_SVM,3,index1=index[b_].tolist())
    # print("rmse_min: {}, b_len: {}".format(rm,rm_j),file=mylog)
    # print(bb,file=mylog)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)),file=mylog)
    ans = "{},{},{},{:.4f}".format(round(len(index)/s_len),m_j,len(b_),round(m / 100, 4) )
    with open("biPLS.txt", "a") as f:
        f.write(ans + "\n")
    mylog.close()
    # print(x)
    # print(np.shape(x))
    # break
    # print(len)
    # print(x0)
def get_iter(x0):
    global bb

    for i in range(len(bb)):
        f = list(bb[0:i])
        f.extend(bb[i + 1:])  # index
        # print(f)
        ans = list(splits[f[0]])
        for v in f[1:]:
            ans.extend(splits[v])  # splits
        xx = []
        for m in x0:
            xx.append(m[ans])
        #  返回原始的index
        yield (ans,np.array(xx))  # x0


def getDataIndex(x0, index):
    l1 = np.shape(x0)[0]
    mm = []
    for i in range(l1):
        mm.append(np.array(x0[i][index]))
    return np.array(mm)

if __name__=='__main__':
    with open("excel.txt", "a") as f:
        f.write("util")
        f.write( "\n\n\n")
    with open("biPLS.txt", "a") as f:
        f.write("util")
        f.write( "\n\n\n")
    # main1(17)
    # main2([2,4,7,9,12,13,21,29,39,40,42,44,47,48,54,58,60,64,72,79,82,84,86,89,90,93,101,104,108,109,110,118,119,120,123,124,125,126,127,128,143,156,157,158,159,173,174,175,176,178,179,180,181,196,198,199,200,201,204,205,210,213,219,224,225,226,239,240,250,251],1)

    llen = [32,26,16,13,9,8,5,4,3,2,1]

    for l in llen:
        main2(s_len=l)

    # for i in range(2,17,2):
    #     main2(s_len=i)