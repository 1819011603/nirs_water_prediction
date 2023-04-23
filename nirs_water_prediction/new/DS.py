import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from nirs.parameters import *

def compute_scale_offset(X, Y):
    # 计算比例因子
    scale = np.std(Y) / np.std(X)
    # 计算偏移量
    offset = np.mean(Y) - scale * np.mean(X)
    return scale, offset


def pds_calibration(X, Y, num_segments=10):
    # 将数据集分成num_segments个子集
    segments_X = np.array_split(X, num_segments)
    segments_Y = np.array_split(Y, num_segments)
    # 初始化比例因子和偏移量数组
    scales = np.zeros(num_segments)
    offsets = np.zeros(num_segments)
    # 计算每个子集的比例因子和偏移量
    for i in range(num_segments):
        scales[i], offsets[i] = compute_scale_offset(segments_X[i], segments_Y[i])
    # 应用比例因子和偏移量
    X_corr = np.zeros_like(X)
    for i in range(num_segments):
        start = i * len(X) // num_segments
        end = (i + 1) * len(X) // num_segments
        X_corr[start:end] = scales[i] * X[start:end] + offsets[i]
    return X_corr





def direct_standardization(X, Y):
    # 计算均值和标准差
    mean_X, std_X = X.mean(), X.std()
    mean_Y, std_Y = Y.mean(), Y.std()

    # 对待校正数据进行标准化处理
    X_std = (X - mean_X) / std_X

    # 将标准化后的待校正数据变换为目标数据的分布
    X_transformed = X_std * std_Y + mean_Y

    return X_transformed


import numpy as np
from sklearn.cross_decomposition import PLSRegression

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def PDS(masterSpectra, slaveSpectra, MWsize, Ncomp, wavelength):
    # Loop Initialization:
    i = MWsize
    k = i - 1
    # Creation of an empty P matrix:
    P = np.zeros((masterSpectra.shape[1], masterSpectra.shape[1] - (2 * i) + 2))
    InterceptReg = []

    while i <= (masterSpectra.shape[1] - k):
        # PLS regression:
        fit = PLSRegression(n_components=Ncomp, scale=False).fit(
            slaveSpectra[:, (i - k)-1:(i + k)-1], masterSpectra[:, i])
        # Extraction of the regression coefficients:
        intercept = fit.y_mean_ - np.dot(fit.x_mean_, fit.coef_)
        coefReg = np.insert(fit.coef_.flatten(), 0, intercept[0])
        InterceptReg.append(coefReg[0])
        coefReg = coefReg[1:]
        # Add coefficients to the transfer matrix:
        P[(i - k)-1:(i + k)-1, i - k] = coefReg
        i += 1

        # Display progression:
        print('\r', round(i / masterSpectra.shape[1] * 100), ' %', sep="", end="")

    P = np.hstack((np.zeros((masterSpectra.shape[1], k)), P, np.zeros((masterSpectra.shape[1], k))))
    InterceptReg = np.hstack((np.zeros(k), np.array(InterceptReg), np.zeros(k)))

    Output = {"P": P, "Intercept": InterceptReg}
    return Output




a,b,c,d = data.get(data_indice[1])
e,f,g,i = data.get(data_indice[5])

Ncomp = 2
MWsize = 2
wavelength = np.arange(a.shape[1])
Master_NIR = a
Slave_NIR = e
# Compute the transfer matrix P:
Pmat = PDS(Master_NIR, Slave_NIR, MWsize, Ncomp, wavelength)
print(Pmat)
# l = PLSRegression(n_components=50)
# l.fit(a,c)
#
# print(r2_score(d,l.predict(b)))
# # e = pds_calibration(e,a,num_segments=8)
# e = direct_standardization(e,a)
# # f = pds_calibration(f,a,num_segments=8)
# f = direct_standardization(f,a)
# l = PLSRegression(n_components=50)
# l.fit(e,g)
# print(r2_score(g,l.predict(e)))
# print(r2_score(i,l.predict(f)))
n = 5
width = a.shape[1] // n
X = a
# 将X分成n段，对每一段进行标准化
scalers = []
for i in range(n):
    scaler = StandardScaler().fit(X[:, i*width:(i+1)*width])
    scalers.append(scaler)

# 对测试数据进行标准化
X_test_scaled = np.empty_like(X)
for i in range(n):
    X_test_scaled[:, i*width:(i+1)*width] = scalers[i].transform(X[:, i*width:(i+1)*width])

model = LinearRegression()

# 训练回归模型并预测
model.fit(X_test_scaled, y)
y_pred = model.predict(X_test_scaled)

# 将预测结果反标准化
y_pred_rescaled = np.empty_like(y_pred)
for i in range(n):
    y_pred_rescaled += scalers[i].inverse_transform(y_pred[:, i*width:(i+1)*width])
y_pred_rescaled /= n
print(y_pred_rescaled)








