import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 生成样本数据
df = pd.read_excel('..\evaluation\棉花产量论文作业的数据.xlsx')
df.head(20)
# 将数据转换为DataFrame
X = df[['种子费', '化肥费', '农药费','机械费','灌溉费']]  # 自变量
# X = df[['机械费','灌溉费']]  # 自变量
Y = df['单产']  # 因变量

# 添加常数项（截距）
X = sm.add_constant(X)

# 定义逐步回归函数
def backward_elimination(X, Y, significance_level=0.05):
    while True:
        model = sm.OLS(Y, X).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            print(f"Removing {excluded_feature} with p-value {max_p_value}")
            X = X.drop(columns=[excluded_feature])
        else:
            break
    return model

# 进行逐步回归
model = backward_elimination(X, Y)
print(model.summary())

# 获取回归系数
intercept = model.params[0]
coef_seed = model.params[1]
coef_fertilizer = model.params[2]

# 创建网格以绘制三维平面
seed_range = np.linspace(X['种子费'].min(), X['种子费'].max(), 100)
fertilizer_range = np.linspace(X['化肥费'].min(), X['化肥费'].max(), 100)
seed_grid, fertilizer_grid = np.meshgrid(seed_range, fertilizer_range)
yield_grid = intercept + coef_seed * seed_grid + coef_fertilizer * fertilizer_grid

# 绘制三维图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制回归平面
surface = ax.plot_surface(seed_grid, fertilizer_grid, yield_grid, cmap='viridis', alpha=0.8)

# 绘制数据点
ax.scatter(X['种子费'], X['化肥费'], Y, color='red', s=50)

# 设置轴标签
ax.set_xlabel('种子费 (X1)')
ax.set_ylabel('化肥费 (X2)')
ax.set_zlabel('单产 (Y)')
ax.set_title('3D Regression Model')

# 添加回归方程
equation = f'Z = {intercept:.2f} + {coef_seed:.4f}*X1 + {coef_fertilizer:.4f}*X2'
ax.text2D(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12)

# 设置视角
ax.view_init(elev=20, azim=120)

# 添加颜色条
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.show()