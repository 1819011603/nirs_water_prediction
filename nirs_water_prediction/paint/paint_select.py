

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
import nirs.util_paint
def line_chart( ypoints,scatter=None, humidity=None):
    """
    线型参数:ls  '‐' 实线，'‐‐' 破折线，'‐.' 点划线，':' 虚线。
    lw linewidth 线宽
    color 颜色

    """
    from scipy.interpolate import make_interp_spline
    # plt.plot(self.xpoints, ypoints, ls='solid', lw=2,label=str(humidity))

    from nirs.parameters import xpoints
    if len(xpoints) < max(scatter):
        xpoints = np.arange(1,len(ypoints)+1)
    xpoints1 = np.array(xpoints,copy=True)
    xpoints2 = np.array(xpoints,copy=True,dtype=str)
    print(len(scatter))
    print("、".join(xpoints2[scatter]) + " nm。")
    print(",".join(xpoints2[scatter]) )
    print(",".join( np.array(scatter,copy=True,dtype=str)) )
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
        plt.plot(X_, Y_, ls='solid', lw=1, label=humidity, color='#c02ab4')

        if scatter is not None:
            y_= X_Y_Spline(xpoints1[scatter])
            plt.scatter(xpoints1[scatter],y_, s=80, marker='s',color="none",edgecolors="blue",label="Selected wavelength")


    else:
        plt.plot(xpoints[len(xpoints ) -len(ypoints):], ypoints, ls='solid', lw=1)
    # 右上角出现小方格  在plt.plot() 定义后plt.legend() 会显示该 label 的内容，否则会报error: No handles with labels found to put in legend.
    if humidity is not  None:
        plt.legend()
    plt.xlabel('Wavelengths(nm)')
    plt.ylabel("Absorbance")

def paint(a):
    from nirs.parameters import X_train, X_test, X_train_copy

    # total = np.concatenate((X_train_copy, X_test), axis=0)

    total = np.mean(X_train, axis=0)


    # a =  [3,5,8,14,26,31,41,43,48,49,53,59,65,72,80,81,85,87,90,110,111,116,120,122,126,127,128,153,158,175,180,184,197,198,202,205,206,214,227,240,241,255]
    print(len(a))
    index = np.array(a)

    plt.figure(figsize=(8, 6), dpi=100)
    # plt.xlim(920, 1620)

    line_chart(total, scatter=index, humidity="Average spectrum")

    from nirs.parameters import xpoints

    suff = 'pdf'
    picture_name = "select"
    dir_path = "./pdf"
    picture_path = get_log_name(picture_name, suff=suff, dir_path=dir_path)
    plt.tight_layout()
    print("save in {}".format(picture_path))
    plt.savefig(picture_path, format='pdf')

    plt.show()


if __name__ == '__main__':
    a = [2, 4, 7, 9, 12, 13, 21, 29, 39, 40, 42, 44, 47, 48, 54, 58, 60, 64, 72, 79, 82, 84, 86, 89, 101, 125, 128, 143,
         156, 157, 175, 176, 180, 181, 199, 200, 201, 204]
    a = [1,6,28,55,57,113,126,165,167,169,177,180,181,229,233]
    paint(a)





