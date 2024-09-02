# 绘制2D散点图,使用多次回归，大致判断变量关系
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel('F:\PythonCodeCangku\CUMCM\Code\evaluation\棉花产量论文作业的数据.xlsx')
X = df['单产']
Y = df['种子费']

# 绘制原始数据的散点图
plt.scatter(X, Y, label='Data Points')

# 一次回归（线性）
slope1, intercept1 = np.polyfit(X, Y, 1)
regression_line1 = slope1 * X + intercept1
plt.plot(X, regression_line1, color='red', label='Linear (1st Degree)')

# =========================================数据点不够密集的话，会因为过拟合等问题从二次回归开始产生多条直线，但其实不是
# 二次回归
coeffs2 = np.polyfit(X, Y, 2)
regression_line2 = np.polyval(coeffs2, X)
plt.plot(X, regression_line2, color='green', label='Quadratic (2nd Degree)')

# 三次回归
coeffs3 = np.polyfit(X, Y, 3)
regression_line3 = np.polyval(coeffs3, X)
plt.plot(X, regression_line3, color='blue', label='Cubic (3rd Degree)')

# 添加标题和标签
plt.title('Scatter Plot with Different Degree Polynomial Fits')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# 显示图表
plt.show()

