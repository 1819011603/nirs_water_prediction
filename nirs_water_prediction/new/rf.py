from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression

from utils import get_log_name


def random_frog_jump_selection(X, y, n_frogs, n_features, max_iter):
    """
    随机蛙跳选择特征算法

    参数：
    X：array-like，shape=(n_samples, n_features)，输入特征
    y：array-like，shape=(n_samples,)，输入目标变量
    n_frogs：int，青蛙数量
    n_features：int，每个青蛙选择的特征数量
    max_iter：int，最大迭代次数

    返回值：
    best_features：array-like，选择的最佳特征
    """
    global best_features, best_fitness
    n_samples, n_all_features = X.shape
    # 生成初始的青蛙
    frogs = np.zeros((n_frogs, n_all_features))
    for i in range(n_frogs):
        features = np.random.choice(n_all_features, n_features, replace=False)
        frogs[i, features] = 1
    f = []
    b = []
    # 循环迭代
    for i in range(max_iter):
        # 计算每个青蛙的适应度
        fitness = []
        for j in range(n_frogs):
            frog_features = frogs[j, :] == 1
            pls = PLSRegression(n_components=1)
            pls.fit(X[:, frog_features], y)
            fitness.append(pls.score(X[:, frog_features], y))

        # 找出适应度最高的青蛙
        best_frog_idx = np.argmax(fitness)
        best_frog_features = frogs[best_frog_idx, :] == 1

        # 计算每个青蛙与最佳青蛙之间的距离
        distances = []
        for j in range(n_frogs):
            frog_features = frogs[j, :] == 1
            if np.sum(frog_features) == 0:
                distances.append(np.inf)
            else:
                # distances.append(np.sum(np.abs(frog_features - best_frog_features)))
                distances.append(np.sum(np.logical_xor(frog_features, best_frog_features)))
        # 找出距离最短的青蛙
        nearest_frog_idx = np.argmax(distances)

        copy_num = 5
        # 青蛙跳跃，从最佳青蛙那里复制一些特征过来
        c = np.where(best_frog_features == True)[0]
        np.random.shuffle(c)
        copy_idx =c [:copy_num]
        frogs[nearest_frog_idx, copy_idx] = 1

        s1 = np.where(nearest_frog_idx == False)[0]
        s2 = np.where(best_frog_features == True)[0]
        np.random.shuffle(s1)

        s3 = np.intersect1d(s1, s2)[:copy_num]
        frogs[nearest_frog_idx, s3] = 0
        # 记录每次迭代最佳的特征集
        if i == 0:
            best_fitness = fitness[best_frog_idx]
            best_features = best_frog_features
        elif fitness[best_frog_idx] > best_fitness:
            best_fitness = fitness[best_frog_idx]
            print(best_fitness)
            best_features = best_frog_features
        f.append(mean(fitness))
        b.append(best_fitness)
    c =  np.where(best_features == True)[0]
    from paint.paint_select import paint
    paint(c)

    plt.figure(figsize=(8, 6), dpi=100)



    plt.plot(np.arange(max_iter),f)
    # plt.plot(np.arange(max_iter),b)
    from nirs.parameters import xpoints

    plt.xlabel("iteration number")
    plt.ylabel("fitness")

    suff = 'pdf'
    picture_name = "select"
    dir_path = "./pdf"
    picture_path = get_log_name(picture_name, suff=suff, dir_path=dir_path)
    plt.legend(["mean fitness"])
    plt.tight_layout()
    print("save in {}".format(picture_path))
    plt.savefig(picture_path, format='pdf')

    plt.show()


    return c


from nirs.parameters import  *

from nirs.nirs_processing import piecewise_polyfit_baseline_correction


c = random_frog_jump_selection(X_train,y_train,1000,50,100)

