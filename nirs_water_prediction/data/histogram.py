from PLS.utils import loadDataSet01




# 画出统计图



x0, y0 = loadDataSet01('train_copy.txt', ', ')

import numpy as np
import matplotlib.pyplot as plt

import random

plt.figure(figsize=(6.4, 4.8), dpi=300)
# 准备数据
x_data = [f"{i*5}-{(i+1)*5-1}" for i in range(1, 16)]
y_data = [0 for i in range(1,16)]

for i in y0:
	y_data[int ((i[0] - 5 )/5)]+=1
x_update = []
y_update = []
for i,item in enumerate(y_data):
	if item != 0:
		x_update.append(x_data[i])
		y_update.append(y_data[i])

x_data = x_update
y_data = y_update

ax = plt.gca()
ax.spines['top'].set_visible(False)
# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 画图，plt.bar()可以画柱状图
for i in range(len(x_data)):
	plt.bar(x_data[i], y_data[i])



#  画图，plt.bar()可以画柱状图
#  设置文字
i = 0
for rect in plt.bar(range((len(x_data))),y_data,color="w",edgecolor="k",width=0.8):
	height = rect.get_height()
	plt.text(rect.get_x()+ rect.get_width()/2, height+0.3,str(y_data[i]),ha="center")
	i+=1
# def normfun(x,mu):
# 	pdf = x / mu
#
# 	return pdf
# import matplotlib.mlab as mlab
# mu, sigma , num_bins = 0, 1, 50
#
# y = normfun(np.array(y_data), np.sum(y_data))
# plt.plot(x_data,y, color='g',linewidth = 3)



# 设置图片名称
plt.title("")
# 设置x轴标签名
plt.xlabel("水分含量")
# 设置y轴标签名
plt.ylabel("频数")

# 显示
plt.show()