


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.random.randint(0, 100, size=100)
y2 = np.random.randint(0, 100, size=100)
y3 = np.random.randint(0, 100, size=100)

# 获取每条曲线的最后一个值
last_value_y1 = y1[-1]
last_value_y2 = y2[-1]
last_value_y3 = y3[-1]

# 创建颜色映射
cmap = plt.cm.get_cmap('Blues')

# 绘制多条曲线并根据最后一个值确定颜色


plt.plot(x, y1, color=cmap(last_value_y1/100), label=f'Last Value: {last_value_y1}')
plt.plot(x, y2, color=cmap(last_value_y2/100), label=f'Last Value: {last_value_y2}')
plt.plot(x, y3, color=cmap(last_value_y3/100), label=f'Last Value: {last_value_y3}')
# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])  # 设置空数组以避免警告
cbar = plt.colorbar(sm)
cbar.set_label('Values')
# 添加图例
plt.legend()

# 显示图形
plt.title("Line Plot with Color based on Last Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
