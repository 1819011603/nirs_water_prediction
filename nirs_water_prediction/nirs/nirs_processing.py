import numpy as np
import pywt
from scipy import signal, sparse
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
# 预处理方法名称全用小写字母


# 多元散射校正
# def msc(data):
#     """"
#     data 横轴方向表示重复测量的数据，纵轴方向表示波段数量
#     """
#     # 计算平均光谱，实际就是x值
#     s_mean = np.mean(data, axis=1)
#
#     # 行列数
#     r, c = data.shape
#
#     # 创建一个单位矩阵
#     msc_x = np.ones((r, c))
#
#     # 遍历各列，实际是各重复测量
#     for i in range(c):
#
#         # y值
#         y = data[:, i]
#
#         # 计算光谱回归系数Ki,Bi
#         lin = LinearRegression()
#         lin.fit(s_mean.reshape(-1, 1), y.reshape(-1, 1))
#         k = lin.coef_
#         b = lin.intercept_
#
#         msc_x[:, i] = (y - b) / k
#
#     return msc_x

# MSC(数据)
def msc(Data):
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

def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return np.array(Di)
# 移动平滑算法（Moving Average Smoothing）
def mas(X, window_size=11):
    # 使用 numpy 的 convolve 函数进行平滑处理
    weights = np.ones(window_size) / window_size
    X_mas = np.apply_along_axis(lambda m: np.convolve(m, weights, mode='same'), axis=1, arr=X)
    return X_mas

# 标准正态变换（Standard Normal Variate Transformation，SNV）
# def snv(data):
#     scaler = StandardScaler(with_mean=True, with_std=True)
#     return scaler.fit_transform(data)
# 标准正态变换

def snv1(data):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(data)

def snv(data):
    m = data.shape[0]
    n = data.shape[1]
    # print(m, n)  #
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return np.array( data_snv)

# 基线偏移校正
def ma(a):
    WSZ = 5
    for i in range(a.shape[0]):
        out0 = np.convolve(a[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(a[i, :-WSZ:-1])[::2] / r)[::-1]
        a[i] = np.concatenate((start, out0, stop))
    return np.array(a)

def boc(X):
    # 对每个光谱进行基线偏移校正
    X_boc = X - X.min(axis=1)[:, np.newaxis]
    # 寻找谷底位置
    valley_indices, _ = find_peaks(-X_boc, prominence=0.1)
    # 寻找谷底位置左右两侧的峰顶
    left_indices, right_indices = [], []
    for i in valley_indices:
        left, right = peak_widths(X_boc[i-10:i+10], [np.argmax(X_boc[i-10:i+10])])[2:]
        left_indices.append(i-10+int(left))
        right_indices.append(i-10+int(right))
    # 使用三次样条插值平滑基线
    for i in range(X_boc.shape[0]):
        X_boc[i, left_indices[i]:right_indices[i]] = np.interp(np.arange(left_indices[i], right_indices[i]),
                                                               [left_indices[i], right_indices[i]],
                                                               [X_boc[i, left_indices[i]], X_boc[i, right_indices[i]]])
    return X_boc



# 基线校正

def piecewise_polyfit_baseline_correction(X):
    X_corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        # 计算每个样本的baseline
        baseline = signal.savgol_filter(X[i, :], 51, 3)
        # 将baseline从原始数据中去除
        X_corrected[i, :] = X[i, :] - baseline
    return X_corrected


def dwt(x0):
    # Daubechies小波系
    wavename = 'db5'
    wavename = 'db38'
    # Symlets小波系
    # print(pywt.wavelist())
    # wavename = 'sym20'
    #
    # # Coiflet小波系
    # wavename = 'coif17'
    # # Biorthogonal小波系
    # wavename = 'bior6.8'

    cA, cD = pywt.dwt(x0, wavename)
    x0 = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
    # x0 = pywt.idwt(cA, None, wavename)  # approximated component
    # x0 = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
    return x0

def dt(X):
    for x in range(len(X)):
        X[x] = signal.detrend(X[x])
    return X

def sg(X):
    for x in range(len(X)):
        X[x] = signal.savgol_filter(X[x], 7, 3, mode="wrap")
    return X
# 最大最小值归一化
def mms(data):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit_transform(data)
def none(X):
    return X
def MA(a, WSZ=5):
    for i in range(a.shape[0]):
        out0 = np.convolve(a[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(a[i, :-WSZ:-1])[::2] / r)[::-1]
        a[i] = np.concatenate((start, out0, stop))
    return np.array(a)
# 一阶导数
def d1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return np.array(Di)

# 二阶导数
def d2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di
class Preprocess:
    def __init__(self, method='none', **kwargs):
        self.method = method   if  not  isinstance(method,str) and hasattr(method, '__iter__') else [method]
        self.params = kwargs

    def transform(self, X, y=None):

        for m in self.method:
            m = m.lower()
            g = globals()


            if m in  g.keys():
                X = g[m](X)
            else:
                raise ValueError('Unsupported preprocess method.')
        return X



