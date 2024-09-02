[[example 1|线性规划例子]]
### 阶梯一般步骤
- 第一步，分析问题，找出决策变量。
- 第二步，根据间题所给条件，找出决策变量必须满足的一组线性等式或者不等式约 束，即为约束条件。
- 第三步，根据问题的目标，构造关于决策变量的一个线性函数，即为目标函数。
有了决策变量、约束条件和目标函数这三个要素之后，一个线性规划模型就建立起来了。
### 线性规划问题的解的概念
可行解: 满足约束条件的解$\vec x=[x_1,x_2,\ldots,x_n]^T$，称为线性规划问题的可行解， 而使目标函数达到最大值的可行解称为最优解。 
可行域: 所有可行解构成的集合称为问题的可行域，记为 R'。
### 灵敏度分析
灵敏度分析是指对系统因周围条件变化显示出来的敏感程度的分析。
在线性规划问题中，都设定 $a_{ij},b_i,c_j$ 是常数，但在许多实际问题中，这些系数往往是 估计值或预测值，经常有少许的变动。
例如在[[example 1|例子1]]中，如果市场条件发生变化,  $c_j$ 值就会随之变化; 生产工艺条件发生改变，会引起 $j_i$ 变化; $a_{ij}$ 也会由于种种原因产生改变。
因此提出以下两个问题:
（1）如果参数a，b，c中的一个或者几个发生了变化，现行最优方案会有什么变化?
（2）将这些参数的变化限制在什么范围内，原最优解仍是最优的?
实际应用中，给定参变量一个步长使其重复求解线性规划问题，以观察最优解的 变化情况，不失为一种可用的数值方法，特别是使用计算机求解时。
需要补充具体方法 #tobesupplemented 
### python代码求解
- **`SciPy`**：简单直接，适合基本的线性规划问题。
- **`PuLP`**：易于使用，专门针对线性规划，支持多种求解器。
- **`cvxpy`**：适合处理更复杂的凸优化问题，灵活性强
假设我们有以下线性规划问题：

$$
最大化 z=2x_1+3x_2
$$
$$
约束条件:\begin{cases}
x_1+x_2\leq4 \\
2x_1+x_2\leq5 \\
x_1,x_2\ge0
\end{cases}
$$
#### 使用 `SciPy` 库
`SciPy` 是一个强大的科学计算库，其中的 `scipy.optimize` 模块提供了 `linprog` 函数，用于解决线性规划问题。
```python
from scipy.optimize import linprog 
# 系数定义 
c = [-2, -3] # 由于是最大化问题，目标函数系数取相反数 
A = [[1, 1], [2, 1]] 
b = [4, 5] 
x0_bounds = (0, None) 
x1_bounds = (0, None) 
# 解决线性规划问题 
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs') 
# 输出结果 
print("最优值:", -res.fun) 
print("最优解:", res.x)
```
#### 使用 `PuLP` 库
`PuLP` 是一个专门用于线性规划问题的库，功能强大且使用简单。
```python
import pulp

# 创建一个线性规划问题（最大化问题）
prob = pulp.LpProblem("Maximize_Z", pulp.LpMaximize)

# 定义决策变量
x1 = pulp.LpVariable('x1', lowBound=0)
x2 = pulp.LpVariable('x2', lowBound=0)

# 目标函数
prob += 2*x1 + 3*x2

# 约束条件
prob += x1 + x2 <= 4
prob += 2*x1 + x2 <= 5

# 求解
prob.solve()

# 输出结果
print("最优值:", pulp.value(prob.objective))
print("最优解:", x1.varValue, x2.varValue)
```
#### 使用 `cvxpy` 库
`cvxpy` 是一个用于凸优化问题的库，可以非常灵活地解决线性规划和其他类型的优化问题。

```python
import cvxpy as cp

# 定义决策变量
x1 = cp.Variable(nonneg=True)
x2 = cp.Variable(nonneg=True)

# 目标函数
objective = cp.Maximize(2*x1 + 3*x2)

# 约束条件
constraints = [
    x1 + x2 <= 4,
    2*x1 + x2 <= 5
]

# 定义并求解问题
problem = cp.Problem(objective, constraints)
problem.solve()

# 输出结果
print("最优值:", problem.value)
print("最优解:", x1.value, x2.value)
```
