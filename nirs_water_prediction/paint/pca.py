import bisect

import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from nirs.parameters import X_train,y_train,X_test,y_test
x_train = X_train
x_train = np.concatenate((x_train,X_test),axis=0)
y_train = np.concatenate((y_train,y_test),axis=0)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"]=["Arial"]
plt.rcParams["font.family"]="Arial"
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
# 复制一份数据集，防止对原始数据进行修改
x_copy = x_train.copy()
y_copy = y_train.copy()

# 定义按照y_train排序的函数
def sortByY(x, y):
    sorted_indices = np.argsort(y)
    return x[sorted_indices], y[sorted_indices]

# 按照y_train的值对x_train和y_train进行排序
x_y = np.column_stack((x_copy, y_copy))
x_y_sorted = sortByY(x_y[:, :-1], x_y[:, -1])
x_train_sorted, y_train_sorted = x_y_sorted

# 使用PCA进行降维
pca = PCA(n_components=3)
x_train_reduced = pca.fit_transform(x_train_sorted)

# 可视化展示
fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')

# 设置数据点的颜色、标记、标签等信息
colors =np.array( ['r', 'g', 'b', 'c', 'm', 'y', 'k'])
markers = np.array(['o', 's', '^', 'v', '*', 'x', 'D'])
start = [0,10, 30, 40, 50, 55, 60, 70,100]

a = []
j = 0

labels = ["Second-Step Drying", "First-Step Drying", "Rolling", "Cooling and breezing", "De-enzyming",
          "Tedding fresh leaving", "Fresh tea"]
# labels.reverse()




for f,i in enumerate( y_train_sorted):
    if i >= start[j+1]:
        j+=1
        ax.scatter(x_train_reduced[f, 0],
                   x_train_reduced[f, 1],
                   x_train_reduced[f, 2],
                   c=colors[j-1], label=labels[j-1],s=60-(7-j)*5)
        print(labels[j-1])
        a.append(j)
        continue
    a.append(j)
    ax.scatter(x_train_reduced[f, 0],
               x_train_reduced[f, 1],
               x_train_reduced[f, 2],
               c=colors[j-1],s=60-(7-j)*5)



ax.set_facecolor('white')


ax.patch.set_facecolor("white")
a = pca.explained_variance_ratio_*100
font = FontProperties()
font.set_weight('bold')
font.set_size(12)
font.set_family('Arial')

ax.set_xlabel('PC1({:.2f}%)'.format(a[0]),fontproperties=font)
ax.set_ylabel('PC2({:.2f}%)'.format(a[1]),fontproperties=font)
ax.set_zlabel('PC3({:.2f}%)'.format(a[2]),fontproperties=font)
# ax.view_init(elev=25, azim=-45)
ax.view_init(elev=25, azim=60)
plt.tight_layout()
ax.legend(loc ="upper left")
# 保存可视化结果
save_path = 'pca_3d_visualization.png'
plt.savefig(save_path,dpi=100)
# plt.show()

print('可视化结果已保存在{}'.format(save_path))