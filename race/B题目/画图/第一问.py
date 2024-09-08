import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# 计算不同m,n下的p值
def calculate_p_values(max_m, max_n, alternative='greater'):
    P = 0.1  # 次品率P
    p_values = np.zeros((max_m, max_n))
    
    for m in range(1, max_m + 1):
        for n in range(0, min(m, max_n)):  # 这里去掉了+1，确保索引不越界
            if alternative == "less":
                p_value = 1 - stats.binom.cdf(n, m, p=P)
            else:  # greater
                p_value = stats.binom.cdf(n - 1, m, p=P)
            p_values[m - 1, n] = p_value
            
    return p_values
# 创建自定义颜色映射

colors = ['#000000', '#2A272A', '#4B4A54', ]
cmap = LinearSegmentedColormap.from_list('custom_gradient', colors)
# 绘制颜色图
def plot_p_values(p_values):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(10, 8))
    plt.imshow(p_values, cmap=cmap, aspect='auto', origin='lower')  # 使用自定义渐变映射
    plt.colorbar(label='p值')  # 图例的标签改为中文
    plt.xlabel('n (不合格次数)')  # x轴标签改为中文
    plt.ylabel('m (样本量)')  # y轴标签改为中文
    plt.title('不同 m 和 n 下的 p 值热力图')  # 标题改为中文
    plt.show()


# 设置最大m和n的值
max_m = 20
max_n = 20

# 计算p值
p_values = calculate_p_values(max_m, max_n, alternative='greater')

# 绘制图像
plot_p_values(p_values)
