
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SimHei"
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
# 定义函数
def func(a):
    return -np.log10(1 + a - np.sqrt(a**2 + 2*a))

# 设置x轴的取值范围
x = np.linspace(0, 1, 1000)

# 计算y轴的值
y = func(x)

# 绘制曲线图
plt.plot(x, y,color ="black")

it = 400
last=-1
y1 =  (y[last]-y[it])/(x[last]-x[it]) * x + y[last]-(y[last]-y[it])/(x[last]-x[it])


ans = -1
t = y1-y
for i in range(1,1000-5):

    if t[i+1]*t[i-1] < 0:
        ans = i
        break




plt.text(x[ans],y[ans]-0.02,"({:.2f},{:.2f})".format(x[ans],y[ans]),color="red")
plt.scatter(x[ans],y[ans],color="red")
plt.plot(x,y1,color="black")


# 添加标题和坐标轴标签
plt.xlim(0.0,1)
plt.xlabel("k/s")
plt.ylabel("吸收度A")

# 显示图形
plt.show()