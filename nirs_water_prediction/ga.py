import time

import numpy

from numpy import mean, power
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# 文档
# https://scikit-opt.github.io/scikit-opt/#/zh/README?id=%e5%ae%89%e8%a3%85


# 介绍
# https://blog.csdn.net/panbaoran913/article/details/128223875



# x_test, y_test = loadDataSet01("./PLS-master/data/test.txt")
# x_test = filter(x_test)

# global x0,y0,x_train, x_test, y_train, y_test,x1,y1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

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

cv = KFold(n_splits=10, shuffle=True, random_state=3)
# a  = FeatureSelection(n_components=15)
# a.fit(x0,y0)
#
# x0 = a.transform(x0)[0]
# x1 = a.transform(x1)[0]
def GA1(C, gamma=None,epsilon=0.1):
    global x0,y0,x1,y1
    # x1 = parameter.pls2.transform(x1)
    # x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=0.20, random_state=13)
    # pls2 = svm.SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    pls2 = MLPRegressor(hidden_layer_sizes=int(C),learning_rate_init=gamma)
    # pls2 = RandomForestRegressor(n_estimators=int(C), max_depth=int(gamma),max_features=int(epsilon),random_state=42)
    # pls2 =AdaBoostRegressor(
    #                 base_estimator=RandomForestRegressor(n_estimators=2, max_depth=11, max_features=int(epsilon),
    #                                                      random_state=42), n_estimators=int(C), random_state=42,
    #                 learning_rate=gamma)
    # pls2.fit(x0, y0)
    # y_predict = pls2.predict(x_test)
    # y1_predict = pls2.predict(x1)
    # return rmse(y_test.ravel(),y_predict.ravel())/100
    return  np.mean(np.sqrt(-cross_val_score(pls2, x0, y0 ,cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)))/100
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
        # pls2 = svm.SVR(kernel='rbf', C=best_x[0], gamma=best_x[1], epsilon=0.1)
        # pls2.fit(x0, y0.ravel())
        # y_predict = pls2.predict(x1)
        # RR, RMSE = getRR_RMSE(y1.ravel(), y_predict)
        # print("RP:{}, RMSEP:{}".format(RR, RMSE))

        paint1(ga.generation_best_Y)

        # import pandas as pd
        #
        # if func.__name__.__contains__('GA'):
        #     Y_history = pd.DataFrame(ga.all_history_Y)
        # else:
        #     Y_history = pd.DataFrame(ga.gbest_y_hist)
        # save_csv(Y_history,get_log_name("GA","csv","./GA/csv"))
        # paintByDataFrame(Y_history)

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
name = ""
def paint1(fitness):
    import matplotlib.pyplot as plt
    fitness = np.array(fitness)
    print(fitness,sep=',')

    a = np.array(fitness,dtype=np.str)
    print(",".join(a))
    global  name

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "SimHei"
    # 解决中文乱码
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    # 解决负号无法显示的问题
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.size'] = 14
    # fitness = np.max(fitness,axis=1)
    # print(fitness)

    fig = plt.figure(figsize = (9,6))
    # 定义适应度数组
    # fitness = [0.6, 0.55, 0.52, 0.48, 0.45, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.38, 0.38]
    # 获取最优适应度
    best_fitness = min(fitness)
    # 获取最优适应度对应的迭代次数
    best_generation =fitness.argmin() + 1

    # 绘制迭代曲线图
    plt.plot(range(1, len(fitness) + 1), fitness, color='black')
    # plt.scatter(best_generation, best_fitness, s=50, marker='*', color='b')
    # plt.text(best_generation + 0.2, best_fitness + 0.02, '最优个体', fontsize=12, color='b')

    # 绘制最小值垂线
    min_line_y = best_fitness
    min_line_x = best_generation
    plt.axvline(x=min_line_x,
                color='black', linestyle='--')
    plt.title('第{}次迭代的RMSECV最低 '.format( best_generation))

    # plt.title('遗传算法进化迭代曲线图')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSECV')
    plt.legend([name])

    pdf_name = get_log_name(name, ".pdf", "./pdf")
    plt.savefig(pdf_name, format='pdf')
    print(pdf_name)
    plt.show()




if __name__ == '__main__':


    # from concurrent.futures import ThreadPoolExecutor
    # theard_pool = ThreadPoolExecutor(4)


    # path = "./GA/csv/GA9_02_27.csv"
    # Y_history=read_csv(path)
    # paintByDataFrame(Y_history)
    # paint()
    name = 'BPNN'
    a = [0.02712343051768199,0.02688756929724469,0.026751547885980297,0.027097263716322442,0.026109784113476562,0.02729413537177048,0.027143014872916878,0.027115484504412733,0.02685184141194278,0.026502669987078235,0.026513477809123307,0.026487168436685707,0.026523016994135763,0.0268334267077329,0.027639495840595742,0.02745503541958881,0.0275663360821796,0.027559061572128386,0.02711672553160711,0.026989071771263,0.026853563922772633,0.026375697977293066,0.02635693177996891,0.02685397500663389,0.026574153804496486,0.02673174198401058,0.027153493350125655,0.02679120989358133,0.02643197357685497,0.02685424868419001,0.026744437594469116,0.026466165272367528,0.026927135741537173,0.026639217802997214,0.026905696956108165,0.026713176864479493,0.025957295746189737,0.02651898230328537,0.026576192517238056,0.026194265767758233,0.026274683061724775,0.02716070915766092,0.0270421494086261,0.02726384747388699,0.026465658596387547,0.02715155675686121,0.027097695316360568,0.0272233807489098,0.02632475336884356,0.02660717601672578,0.027323549397425338,0.02629333180036288,0.02669366648862563,0.026956791397994614,0.027249941712740425,0.02611044350186421,0.026886078477687526,0.02673656135133104,0.026456078709002448,0.026707666911898822,0.026854322915808325,0.026689798866379396,0.02623676491519233,0.026721566936346596,0.026502554676527196,0.02657429309266937,0.026071485932729052,0.027178355369741468,0.025990075155157517,0.026832647008607217,0.026778899002530308,0.027023914104677797,0.026593314669788582,0.02654419584337413,0.026957999978644696,0.026597767281917265,0.02691921471115401,0.027375948473446095,0.026380864277233504,0.02681637526277625,0.026504002658392763,0.02691815841537547,0.026402850029508717,0.025898525434287386,0.026303182406141382,0.026348862546637392,0.026192299098576823,0.026403858643548515,0.026963590545429877,0.026706874151113474,0.026577090661633468,0.027318691078367915,0.026584298838875683,0.02681819445602485,0.026849927047892005,0.02661057192355968,0.025970010740501204,0.026584303513488273,0.026098746206787534,0.027066523971217435]
    paint1(np.array(a)-0.015)

    # 遗传算法
    from sko.GA import GA
    # [883.0056267   48.46402141]   0.9899132355433415
    # 312.69434992  62.84171658 0.9908128599175112
    # theard_pool.submit(main, GA(func=GA1, n_dim=2, size_pop=50, max_iter=300, prob_mut=0.001, lb=[1, 1], ub=[2000, 1000], precision=1e-6))
    name = "BPNN"
    # main(GA(func=GA1, n_dim=3, size_pop=10, max_iter=100, prob_mut=0.02, lb=[2, 1 ,0.00001], ub=[100, 5,1], precision=1e-4))
    # main(GA(func=GA1, n_dim=2, size_pop=20, max_iter=100, prob_mut=0.02, lb=[20,0.01], ub=[120,0.2], precision=1e-4))



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
    # P = PSO(func=GA1, n_dim=3, pop=50, max_iter=50, lb=[0.00000001, 0.00000001,0.00000001], ub=[10000, 10000,10000], w=0.6, c1=0.5, c2=0.5,verbose=True)
    # # P.paint_w_cp_cg_curse()
    # print("改进")
    # main(P)
