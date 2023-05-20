import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SimHei"
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False

plt.rcParams['font.size'] = 16
# 定义函数
def f(x, k):
    return (2*np.exp(-k*x))/(1 + np.exp(-k*x))

# 生成 x 坐标轴数据
x = np.linspace(0, 10, 1000)

# 绘制函数图像
# plt.plot(x, f(x, k=0.1), label='k=0.1')
plt.plot(x, f(x, k=0.5), label='k=0.5')
plt.plot(x, f(x, k=1), label='k=1')
plt.plot(x, f(x, k=2), label='k=2')
plt.plot(x, f(x, k=3), label='k=3')
plt.plot(x, f(x, k=4), label='k=4')

plt.xlabel("$\partial$")
plt.ylabel("y")
plt.xlim(0,10)
plt.ylim(0,1)
# 添加图例和标题
plt.legend()
# plt.title(r'$y = \frac{1 - e^{-k\cdot\partial}}{1 + e^{-k\cdot\partial}}$')
plt.title(r'$y = \frac{2 \cdot e^{-k\cdot\partial}}{1 + e^{-k\cdot\partial}}$')


plt.savefig("Pm.pdf")

from nirs.parameters import *
saveWMF("pm.WMF")
# 显示图形
plt.show()