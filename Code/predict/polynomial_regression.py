import numpy as np
import matplotlib.pyplot as plt

# 1. 创建数据集
# 生成 X 数据（从 -10 到 10 的均匀分布）
x = np.linspace(-10, 10, 100)

# 生成 Y 数据，假设多项式为 y = a*x^3 + b*x^2 + c*x + d，并添加一些随机噪声
a = 0.5
b = -1.5
c = 2
d = 3
noise = np.random.normal(0, 10, size=x.shape)
y = a * x**3 + b * x**2 + c * x + d + noise

# 2. 使用 numpy.polyfit 进行多项式回归
# 这里我们拟合一个2次多项式（即二次曲线）
degree = 3
coefficients = np.polyfit(x, y, degree)

# 3. np.polyfit 返回的是多项式的系数
# 对于二次多项式 ax^2 + bx + c，系数为 [a, b, c]
print("Polynomial coefficients:", coefficients)

# 4. 使用 numpy.poly1d 创建多项式函数
# poly1d 函数会创建一个可以计算多项式值的函数对象
polynomial = np.poly1d(coefficients)

# 5. 可视化原始数据和拟合曲线
# 生成100个在x范围内的点，用于绘制平滑的曲线
x_fit = np.linspace(min(x), max(x), 100)
y_fit = polynomial(x_fit)

# 6. 绘制原始数据点和拟合的多项式曲线
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_fit, y_fit, color='blue', label=f'{degree} Degree Polynomial Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
