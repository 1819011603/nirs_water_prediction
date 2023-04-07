import numpy as np
import pywt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
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
def piecewise_polyfit_baseline_correction(X, window_size=21, order=3, num_segments=10):
    # 将每个光谱分为num_segments个段
    num_points = X.shape[1]
    segment_size = int(np.ceil(num_points / num_segments))
    left_edges = np.arange(0, num_points, segment_size)
    right_edges = left_edges + segment_size - 1
    right_edges[-1] = num_points - 1

    # 对每个段进行多项式拟合
    for i in range(num_segments):
        left_edge = left_edges[i]
        right_edge = right_edges[i]
        segment_X = X[:, left_edge:right_edge + 1]
        segment_indices = np.arange(left_edge, right_edge + 1)
        for j in range(segment_X.shape[0]):
            segment_X[j] = savgol_filter(segment_X[j], window_size, order)
        segment_baseline = np.mean(segment_X, axis=0)
        X[:, left_edge:right_edge + 1] = X[:, left_edge:right_edge + 1] - segment_baseline

    return X




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



