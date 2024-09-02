import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可重复
np.random.seed(42)

# 生成 X 数据（从 -10 到 10 的均匀分布）
X = np.linspace(-10, 10, 100)

# 生成 Y 数据，假设多项式为 y = a*x^3 + b*x^2 + c*x + d，并添加一些随机噪声
a = 0.5
b = -1.5
c = 2
d = 3
noise = np.random.normal(0, 10, size=X.shape)
Y = a * X**3 + b * X**2 + c * X + d + noise

# 将数据保存为 pandas DataFrame
df = pd.DataFrame({'X': X, 'Y': Y})

# 绘制原始数据的散点图
plt.scatter(X, Y, label='Data Points')

# 一次回归（线性）
slope1, intercept1 = np.polyfit(X, Y, 1)
regression_line1 = slope1 * X + intercept1
plt.plot(X, regression_line1, color='red', label='Linear (1st Degree)')

# 二次回归
coeffs2 = np.polyfit(X, Y, 2)
regression_line2 = np.polyval(coeffs2, X)
plt.plot(X, regression_line2, color='green', label='Quadratic (2nd Degree)')

# 三次回归
coeffs3 = np.polyfit(X, Y, 3)
regression_line3 = np.polyval(coeffs3, X)
plt.plot(X, regression_line3, color='pink', label='Cubic (3rd Degree)')

# 添加标题和标签
plt.title('Scatter Plot with Different Degree Polynomial Fits')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# 显示图表
plt.show()