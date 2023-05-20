
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SimHei"
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
plt.rcParams['font.size'] = 15


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from nirs.parameters import X_train
fig = plt.figure(figsize=(8,8))
# 进行PCA降维计算，假设我们要降到3个维度
pca = PCA(n_components=15)
pca.fit(X_train)

# 计算每个主成分的方差比例和累积方差比例
variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(variance_ratio)

# 绘制方差比例和累积方差比例的柱状图
fig, ax = plt.subplots()
ax.bar(range(len(variance_ratio)), variance_ratio, label='主成分方差比例')
ax.plot(range(len(cumulative_variance_ratio)), cumulative_variance_ratio, label='累计方差贡献率', c='r')
ax.set_xlabel('主成分数')
ax.set_ylabel('方差比例')
ax.set_xticks(range(len(variance_ratio)))
ax.set_xticklabels(['{}'.format(x+1) for x in range(len(variance_ratio))])
ax.legend()
plt.savefig("方差比例.pdf")
plt.show()
