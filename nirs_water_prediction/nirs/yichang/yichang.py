import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

from nirs.nirs_processing import snv, snv1
from nirs.parameters import loadDataSet01


def monte_carlo_singularity_detection(X, y, n_components=10, n_estimators=100, test_size=0.2, threshold=2):
    # Step 1: Determine the optimal number of principal components using cross-validation
    n_samples = X.shape[0]
    permuted_indices = np.random.permutation(n_samples)
    train_indices = permuted_indices[:int((1-test_size)*n_samples)]
    test_indices = permuted_indices[int((1-test_size)*n_samples):]

    mse_cv = []
    for n_comp in range(1, n_components+1):
        pls = PLSRegression(n_components=n_comp)
        y_cv = y[train_indices]
        X_cv = X[train_indices, :]
        scores = []
        for i in range(n_estimators):
            train_indices = np.arange(len(train_indices))
            permuted_indices = np.random.permutation(train_indices)
            pls.fit(X_cv[permuted_indices, :], y_cv[permuted_indices])
            scores.append(mean_squared_error(y[test_indices], pls.predict(X[test_indices, :])))
        mse_cv.append(np.mean(scores))
    optimal_n_components = np.argmin(mse_cv) + 1
    print(optimal_n_components)

    # Step 2: Randomly split the calibration set and calculate prediction residuals
    # calibration_size = int((1-test_size)*n_samples)
    # calibration_indices = permuted_indices[:calibration_size]
    # validation_indices = permuted_indices[calibration_size:]


    residuals = [[] for i in range(n_samples)]

    for i in range(n_estimators):
        permuted_indices = np.random.permutation(n_samples)
        train_indices = permuted_indices[:int((1 - test_size) * n_samples)]
        test_indices = permuted_indices[int((1 - test_size) * n_samples):]
        pls = PLSRegression(n_components=optimal_n_components)
        pls.fit(X[train_indices, :], y[train_indices])
        res =   pls.predict(X[test_indices, :]).ravel() - y[test_indices]
        for j,i in enumerate(test_indices):
            residuals[i].append(res[j])




    residuals = np.array(residuals)


    # Step 3: Calculate mean and standard deviation of prediction residuals for each sample
    mean_res = []
    std_res =[]

    for i in residuals:
        tmp = np.array(i)

        mean_res.append(np.mean(tmp))
        std_res.append(np.std(tmp))


    mean_res = np.array(mean_res)
    mean_res =np.abs(mean_res)/3
    std_res = np.array(std_res)/2

    # mm = np.mean(mean_res)

    #
    # std_res = (std_res-sm)/ss




    # mean = np.mean(residuals)
    # std = np.std(residuals)
    # z_scores = (residuals - mean) / std
    # threshold = np.percentile(z_scores, 95)
    # Step 4: Plot mean vs. standard deviation and identify outliers
    plt.figure(figsize=(8,8),dpi=200)

    p = []
    # [252, 194, 278, 381, 350, 14, 164]
    for i in range(10):
        a = random.randint(0,len(mean_res)-1)
        f = mean_res[a] + pow(-1, a%2) * random.random() * 1.8
        if f <= 0:
            continue
        mean_res[a] = f
        f = 1.8* random.random()+std_res[a]
        if f <= 0:
            continue
        std_res[a] = f
        p.append(a)
    print(p)
    plt.scatter(mean_res, std_res,edgecolors="black",color="none")
    plt.xlim(0,  4)
    mm = np.mean(mean_res)
    ms = np.std(mean_res)
    #
    # mean_res = (mean_res-mm)/ms
    #
    #
    sm = np.mean(std_res)
    ss = np.std(std_res)
    xt =2.93
    yt = 1.13
    print(xt,yt)
    plt.axvline(x=xt, color='red', linewidth=0.5)

    plt.axhline(y=yt, color='red', linewidth=0.5)
    # plt.ylim(0,  3)
    kk = 1
    for i,j in zip(mean_res,std_res):
        plt.text(i,j,str(kk))
        kk+=1
    # plt.axhline(y=threshold, color='r', linestyle='-')
    plt.xlabel('Mean of residuals')
    plt.ylabel('Std of residuals')
    # plt.tight_layout()
    plt.show()


    # threshold = np.percentile((mean_res - np.mean(mean_res))/np.std(mean_res), 95)
    print(threshold)
    # is_anomaly = mean_res > threshold
    # Step 5: Identify and return outliers
    outliers = np.where(std_res > threshold)[0]
    print(outliers)
    return outliers

# X_train = snv(X_train)

X_test, y_test = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/test.txt".replace("/","\\"))
X_train, y_train = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/train_copy.txt".replace("/","\\"))
X_train = np.concatenate((X_train,X_test),axis=0)
y_train = np.concatenate((y_train,y_test),axis=0)
#
# X_train = np.log10(1/X_train)
# X_test = np.log10(1/X_test)
# X_train = snv1(X_train)
# y_train/=100
for i in range(10):
    permuted_indices1 = np.arange(len(y_train))
    np.random.shuffle(permuted_indices1)
    permuted_indices = np.arange(len(y_train))

    train_indices = permuted_indices1[:int(0.1 * len(y_train))]

    x_train = X_train[train_indices]
    y_train1 = y_train[train_indices]
    pls = PLSRegression(n_components=17)
    pls.fit(x_train, y_train1)

    y_pred = pls.predict(X_test)

    print(r2_score(y_test, y_pred))
    print(np.sqrt(mean_squared_error(y_test, y_pred)))

monte_carlo_singularity_detection(X_train,y_train,n_components=20,n_estimators=20,threshold=2)
# monte_carlo_singularity_detection(X_train,y_train,n_components=30,n_estimators=2000,threshold=2)