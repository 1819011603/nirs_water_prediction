import numpy

from nirs.parameters import X_train,y_train
from paint.paint_select import line_chart
import nirs.util_paint
X = X_train
y = y_train/100
import SPA

spa = SPA.SPA()
# 数据归一化
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 这里feature_range根据需要自行设置，默认（0,1）

X_ = min_max_scaler.fit_transform(X)


# 建模集测试集分割
from sklearn.model_selection import train_test_split

# 注意 X_ 分割
# 若存在 运行后出现 波段选择为 最小值 可适当调整 建模集与测试集比例 test_size 值 0.3 - 0.5
Xcal, Xval, ycal, yval = train_test_split(X_, y, test_size=0.4, random_state=3)

var_sel, var_sel_phase2 = spa.spa(
        Xcal, ycal, m_min=33, m_max=50,Xval=Xval, yval=yval, autoscaling=1)


print(",".join(numpy.array(var_sel,dtype=str)))
import numpy as np
from nirs.parameters import X_train, X_test, X_train_copy
index = np.array(var_sel)
index.sort()
total = np.concatenate((X_train_copy, X_test), axis=0)

total = np.mean(total,axis=0)

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
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["font.family"]="SimHei"
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(8, 6), dpi=100)
plt.xlim(920, 1620)
line_chart(total,scatter=index,humidity="Average spectrum")


suff = 'pdf'
picture_name = "spa"
dir_path = "./pdf"
picture_path = get_log_name(picture_name, suff=suff, dir_path=dir_path)
plt.tight_layout()
print("save in {}".format(picture_path))
plt.savefig(picture_path, format='pdf')

plt.show()