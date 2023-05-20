


import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
plt.rcParams['font.size']=14

def paintxiang(data,labels,ylabel="$R^2$"):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    # sns.set(style="whitegrid")
    ax = sns.boxplot(data=data, showfliers=False,palette="Blues", orient="v", width=0.3, showmeans=True,
                     meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"})
    ax.set_xticklabels(labels)

    # 添加中位线的标记
    means = [np.mean(d) for d in data]
    means_labels = [str(np.round(m, 2)) for m in means]

    medians = [np.mean(d) for d in data]
    median_labels = [str(np.round(m, 2)) for m in medians]

    pos = range(len(data))
    for tick, label in zip(pos, ax.get_xticklabels()):
        # ax.text(pos[tick], medians[tick] + 0.15, median_labels[tick], horizontalalignment='center', size='medium',
        #         color='black', weight='semibold')
        ax.text(pos[tick] + 0.3, means[tick] - 0.1, means_labels[tick], horizontalalignment='center', size='medium',
                color='black', weight='semibold')


    # 设置图例
    plt.plot([-10], [0], color='black', label='Median Value')
    plt.plot([-10], [0], marker='s', color='w', markerfacecolor='white', markersize=8, markeredgecolor='black',
             label='Mean Value')

    plt.plot([-10], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, markeredgecolor='black',
             label='25%-75%')

    plt.xlim(-0.5, len(labels) - 0.5)
    plt.legend()
    # 显示图形
    # plt.title("Box Plot with Mean")
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.show()
