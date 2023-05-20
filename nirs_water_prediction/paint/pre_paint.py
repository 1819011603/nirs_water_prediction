import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.ndimage.filters
from matplotlib.ticker import MultipleLocator
from scipy import signal
import torch
import torch.nn as nn

from nirs.nirs_processing import Preprocess
from utils import get_log_name

# plt.rcParams["font.family"] = "SimHei"
# 解决中文乱码
plt.rcParams["font.sans-serif"]=["Arial"]
plt.rcParams["font.family"]="Arial"
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False

from nirs.util_paint import *

plt.rcParams['font.size']=20

import nirs.util_paint
def line_chart( ypoints, humidity=None,cmp = None):
    """
    线型参数:ls  '‐' 实线，'‐‐' 破折线，'‐.' 点划线，':' 虚线。
    lw linewidth 线宽
    color 颜色

    """
    from scipy.interpolate import make_interp_spline
    # plt.plot(self.xpoints, ypoints, ls='solid', lw=2,label=str(humidity))

    from nirs.parameters import xpoints

    # if(len(xpoints) != len(ypoints)):
    #     xpoints = np.arange(len(ypoints))

    x = xpoints
    y = ypoints
    # print(min(x),max(x))
    # plt.plot(x, y, ls='solid', lw=1)
    # m1, m2 = np.min(x), np.max(x)
    # ma1, ma2 = np.min(y), np.max(y)
    # plt.xlim(m1, m2)
    # plt.ylim(ma1, ma2)


    # plt.xlim(m1,m2)
    # plt.ylim(ma1,ma2)
    if(len(ypoints) == len(xpoints)):
        sep = 15

        # ys = []
        # for i in range(int(len(xpoints)/sep)):
        #     start = i*sep
        #     end = (i+1)*sep
        #     ys.append(np.mean(ypoints[start:end]))
        #     if i == int(len(xpoints)/sep)-1:
        #         ys.append(np.mean(ypoints[end:]))
        # ypoints = np.array(ys)
        xpoints = xpoints[::sep]
        ypoints = ypoints[::sep]
        # 平滑
        X_Y_Spline = make_interp_spline(xpoints, ypoints)
        X_ = np.linspace(min(xpoints), max(xpoints), 1000)
        Y_ = X_Y_Spline(X_)
        if humidity is not  None:
            plt.plot(X_, Y_, ls='solid', lw=1 , color=cmp(humidity/100))
        else:
            plt.plot(X_, Y_, ls='solid', lw=1)
    else:
        plt.plot(xpoints[len(xpoints ) -len(ypoints):], ypoints, ls='solid', lw=1)
    # 右上角出现小方格  在plt.plot() 定义后plt.legend() 会显示该 label 的内容，否则会报error: No handles with labels found to put in legend.
    if humidity is not  None:
        # plt.legend()
        pass
    plt.xlabel('Wavelengths(nm)')
    plt.ylabel("Absorbance")

def paint(X,pre,y):

    plt.figure(figsize=(12,7))
    plt.xlim(920,1620)
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    c = "viridis"
    c = "Blues"
    c = "cool"
    c = "Set1"
    c = "tab20"
    c = "Set3"
    c = "tab10"
    c = "Set2"
    c = "Blues_r"
    cmap = plt.cm.get_cmap(c)
    for i,tea in enumerate(X):

        line_chart(tea,y[i],cmap)
    # t.wiener_filtering()

    # sm = plt.cm.ScalarMappable(cmap=cmap)
    sm =  plt.cm.ScalarMappable(cmap=c, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])  # 设置空数组以避免警告
    cbar = plt.colorbar(sm)
    cbar.set_label('Moisture Content(%)')
    suff='pdf'
    picture_name=pre
    dir_path="./pre1"
    picture_path = get_log_name(picture_name, suff=suff, dir_path=dir_path)
    plt.tight_layout()
    print("save in {}".format(picture_path))
    plt.savefig(picture_path,dpi=50)
    plt.savefig( get_log_name(picture_name, suff="png", dir_path=dir_path),dpi=100)
    plt.show()

if __name__ == '__main__':
    from nirs.parameters import X_train_copy, X_test, preprocess_args,y_train_copy,y_test

    total = np.concatenate((X_train_copy,X_test),axis=0)
    total_y = np.concatenate((y_train_copy,y_test),axis=0)
    abo =total

    # preprocess = [["none"], ["SNV"], ["MSC"], ["SG"], ["DT"], ["MSC", "DT"], ["SG", "DT"], ["DT", "DT"]]
    # 多元散射校正（MSC）、标准正态变换(SNV)、离散小波变换(DWT)、一阶导数、正交信号校正(orthogonal signal correction， OSC)、Savitzky-Golay(S-G)平滑滤波和去趋势(Detrend)
    preprocess = [["msc"], ["SNV"], ["dwt"], ["d1"],['ma'], ["piecewise_polyfit_baseline_correction"], ["sg"], ["dt"]]
    # preprocess = [ ["piecewise_polyfit_baseline_correction"]]
    preprocess = [["d1"]]
    preprocess = [["none"]]
    preprocess = [["sg","dt"]]

    for i,item in enumerate(total_y):
        total_y[i] += np.random.randint(-3,3)

    for p in preprocess:
        preprocess_args["method"] = p

        preprocessor = Preprocess(**preprocess_args)
        res = preprocessor.transform(np.array(abo,copy=True))
        paint(res,"_".join(p),total_y)


