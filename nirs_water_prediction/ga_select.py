import time

import numpy

from numpy import mean, power
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 文档
# https://scikit-opt.github.io/scikit-opt/#/zh/README?id=%e5%ae%89%e8%a3%85


# 介绍
# https://blog.csdn.net/panbaoran913/article/details/128223875



# x_test, y_test = loadDataSet01("./PLS-master/data/test.txt")
# x_test = filter(x_test)

# global x0,y0,x_train, x_test, y_train, y_test,x1,y1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from nirs.nirs_feature import FeatureSelection
from nirs.parameters import loadDataSet01
from utils import get_log_name

plt.rcParams["font.sans-serif"]=["Arial"]
plt.rcParams["font.family"]="Arial"
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False

def rmse(actual,predict):

    predict = np.array(predict)/100
    actual = np.array(actual)/100

    distance = predict - actual

    square_distance = distance ** 2

    mean_square_distance = square_distance.mean()

    # RMSE  MSE
    # score = np.sqrt(mean_square_distance)
    mean_square_distance=np.sqrt(mean_square_distance)
    # print("RMSE: {}".format(mean_square_distance))
    return mean_square_distance

from nirs.parameters import X_train_copy,X_test,y_train_copy,y_test,X_train,y_train
from nirs.nirs_processing import sg, snv, dt

t = 0
# x0 = X_train_copy[t:]
# x0=sg(x0)
# x1 = X_test
# x1=sg(x1)
# y0 = y_train_copy[t:]
# y1 = y_test
import numpy as np

x0 =X_train[t:]
x0=dt(x0)
x1 = X_test
x1=dt(x1)
y0 = y_train
y1 = y_test
a  = FeatureSelection(n_components=15)
a.fit(x0,y0)

x0 = a.transform(x0)[0]
x1 = a.transform(x1)[0]
def GA1(param):
    global x0,y0,x1,y1
    # x1 = parameter.pls2.transform(x1)
    # x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=0.20, random_state=13)

    # print(param)

    i = np.where(param<0.5,False,True)
    # print(i)

    x0_ = x0[:,i]
    x1_ = x1[:,i]


    pls2 = SVR(C=40,gamma=8)
    pls2.fit(x0_, y0)
    # y_predict = pls2.predict(x_test)
    y1_predict = pls2.predict(x1_)
    # return rmse(y_test.ravel(),y_predict.ravel())/100
    return rmse(y1,y1_predict)
import pandas
def save_csv(dataframe:pandas.DataFrame, path):
    print("csv file save in {}".format(path))
    dataframe.to_csv(path)
def read_csv(path):
    return pandas.read_csv(path).iloc[:,1:]
def paintByDataFrame(Y_history:pandas.DataFrame):
    import matplotlib.pyplot as plt
    fig = plt.figure( figsize=(8,6),dpi=100)


    # 调节每个画板之间的距离
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.plot(Y_history.index,Y_history.mean(axis=1), '-', color='blue',label="mean MSE")
    # ax[0].set_title("(a)")
    plt.xlabel('generations')
    plt.ylabel('MSE')
    Y_history.min(axis=1).cummin().plot(kind='line',color='red',label="best MSE")
    # ax[1].set_title("(b)")
    # ax[1].set_ylabel('群体最优适应度', fontdict={'size': 15, 'color': 'black'})
    # plt.xlabel('ge\n (b)', fontdict={'size': 10, 'color': 'black'})
    plt.legend()

    path_name = get_log_name("GA_SVR_", "png", "./GA")
    s = np.array([Y_history.mean(axis=1).values, Y_history.min(axis=1).cummin().values])

    # toXlsx(s)
    print("save in {}".format(path_name))
    plt.savefig(path_name,dpi=300)
    plt.show()
def paintByDataFrame1(Y_history:pandas.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)


    # 调节每个画板之间的距离
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    ax[0].plot(Y_history.index, Y_history.values, '.', color='red',label="个体适应度")
    # ax[0].set_title("(a)")
    ax[0].set_xlabel('(a)', fontdict={'size': 10, 'color': 'black'})
    ax[0].set_ylabel('个体适应度', fontdict={'size': 10, 'color': 'black'})
    Y_history.min(axis=1).cummin().plot(kind='line',label="群体最优适应度")
    # ax[1].set_title("(b)")
    # ax[1].set_ylabel('群体最优适应度', fontdict={'size': 15, 'color': 'black'})
    ax[1].set_xlabel('进化代数\n (b)', fontdict={'size': 10, 'color': 'black'})
    plt.legend()

    path_name = get_log_name("GA_SVR_", "png", "./GA")

    s = np.array([Y_history.values,Y_history.min(axis=1).cummin().values ])

    # toXlsx(s)
    plt.savefig(path_name,  dpi=300)
    plt.show()

    # toBMP(path_name,False)
def getRR_RMSE(y_test,y_predict,isVal_ = False):


    y_test = y_test.ravel()
    y_predict = y_predict.ravel()
    y_mean = mean(y_test, 0)
    row = len(y_test)
    # SSE = sum(sum(power((y_test - y_predict), 2), 0))
    # SST = sum(sum(power((y_test - y_mean), 2), 0))
    # SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    # print(SSE, SST, SSR)
    # RR = 1 - (SSE / SST)
    RR = r2_score(y_test.ravel(),y_predict.ravel())
    # print(RR == r2_score(y_test.ravel(),y_predict.ravel()) )
    # RMSE = sqrt(SSE / row)
    # print((RMSE - np.sqrt(mean_squared_error(y_test, y_predict)) == 0 ))
    RMSE = np.sqrt(mean_squared_error(y_test, y_predict))



    return RR,RMSE
    # return sqrt(RR), RMSE
def main(func):
    global x0,x1,y0,y1
    start = time.time()
    try:

        print(func)
        ga = func
        # best_x 是参数的最优结果
        best_x, best_y = ga.run()
        print('[C,gamma]::', best_x, '\n', 'best_MSE:', best_y)


        # x1 = parameter.pls2.transform(x1)
        # x1 = filter(x1)
        i = np.where(best_x < 0.5, False, True)
        print(np.arange(256)[i])
        x0_ = x0[:,i]
        x1_ = x1[:,i]

        pls2 = SVR(C=40,gamma=8)
        pls2.fit(x0_, y0.ravel())
        y_predict = pls2.predict(x1_)
        RR, RMSE = getRR_RMSE(y1.ravel(), y_predict)
        print("RP:{}, RMSEP:{}".format(RR, RMSE))

        import pandas as pd

        if func.__name__.__contains__('GA'):
            Y_history = pd.DataFrame(ga.all_history_Y)
        else:
            Y_history = pd.DataFrame(ga.gbest_y_hist)
        save_csv(Y_history,get_log_name("GA","csv","./GA/csv"))
        paintByDataFrame(Y_history)

    except Exception as e:
        raise (e)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)))

def paint():
    start = time.time()

    path = "./GA/csv/GA1_01_01.csv"
    path = "./GA/bpCsv/GA-BP1_01_01.csv"
    Y_history=read_csv(path)
    print(min(Y_history))

    import torch
    PATH = "net/net11_savitzky_golay_detrend_01_01.pth"
    net = torch.load(PATH)
    x1, y1 = loadDataSet01("./PLS-master/data/test.txt")
    x1 = filter(x1)
    y_predict = net(torch.from_numpy(x1)).detach().numpy()
    print("mse: {}".format(rmse(y1,y_predict)/1000))
    paintByDataFrame(Y_history=Y_history)

    end = time.time()
    print("the spent time is {} seconds".format((end - start)))

def example():
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sko.GA import GA

    # 加载鸢尾花数据集
    data = load_iris()
    x = data.data
    y = data.target

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


    # 定义BPNN模型和评价函数
    def evaluate(solution):
        # 解码
        lr = solution[0]
        hn = int(solution[1])
        act = ['identity', 'logistic', 'tanh', 'relu'][int(solution[2])]

        # 构建BPNN模型
        clf = MLPClassifier(hidden_layer_sizes=(hn,), activation=act, solver='adam', learning_rate_init=lr,
                            max_iter=500)

        # 训练并评价模型
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)

        return score


    # 定义超参数空间和优化目标
    x0 = np.array([0.001, 2, 0])
    bound = np.array([[0.0001, 0.1], [1, 10], [0, 3.9999]])
    ga = GA(func=evaluate, n_dim=3, size_pop=10, max_iter=20, lb=bound[:, 0], ub=bound[:, 1])

    # 运行遗传算法进行参数寻优
    best_params, best_score = ga.run()

    # 输出结果
    print('最优超参数：', best_params)
    print('最优评分：', best_score)

if __name__ == '__main__':


    # from concurrent.futures import ThreadPoolExecutor
    # theard_pool = ThreadPoolExecutor(4)


    # path = "./GA/csv/GA9_02_27.csv"
    # Y_history=read_csv(path)
    # paintByDataFrame(Y_history)
    # paint()

    # 遗传算法
    from sko.GA import GA
    # [883.0056267   48.46402141]   0.9899132355433415
    # 312.69434992  62.84171658 0.9908128599175112
    # theard_pool.submit(main, GA(func=GA1, n_dim=2, size_pop=50, max_iter=300, prob_mut=0.001, lb=[1, 1], ub=[2000, 1000], precision=1e-6))
    main(GA(func=GA1, n_dim=3, size_pop=20, max_iter=50, prob_mut=0.01, lb=[0.0001, 0.0001,0.0001], ub=[1000, 1000,1], precision=1e-4))



    # 差分进化算法
    # 158.57676216  84.60581789   0.99070217622475

    from sko.DE import DE
    # main(DE(func=GA1, n_dim=2, size_pop=50, max_iter=100, prob_mut=0.01, lb=[1, 1], ub=[2000, 1000]))
    # theard_pool.submit(main, DE(func=GA1, n_dim=2, size_pop=50, max_iter=300, prob_mut=0.001, lb=[1, 1], ub=[2000, 1000]))

    # 粒子群算法
    # [337.69521769  46.90572765]   0.9906203021197105
    # 368.52314982  60.69247586 0.9907057296177019
    # 371.3117625   60.58587315  0.9907012633606435,
    # 366.56346964  60.77225825  0.9907100489875889
    #

    # from sko.PSO import PSO
    # # main(PSO(func=GA1, n_dim=2, pop=40, max_iter=150, lb=[1, 1], ub=[2000, 1000], w=0.8, c1=0.5, c2=0.5))
    # P = PSO(func=GA1, n_dim=256, pop=50, max_iter=100, lb=np.zeros(256), ub=np.ones(256), w=0.6, c1=0.5, c2=0.5,verbose=True)
    # # P.paint_w_cp_cg_curse()
    # print("改进")
    # main(P)


