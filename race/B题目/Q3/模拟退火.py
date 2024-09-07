import numpy as np


def generate_random_strategy():
    # 假设有 8 个 x 变量和 3 个 k, z 变量
    strategy = []
    
    # 随机生成 8 个 x 变量 (0 或 1)
    for i in range(1, 9):
        x = np.random.choice([0, 1])
        strategy.append(x)
      
    # 随机生成 3 个 k 变量 (0 或 1)
    for i in range(1, 4):
        y = np.random.choice([0, 1])
        strategy.append(y)

    # 随机生成 3 个 k 变量 (0 或 1)
    for i in range(1, 4):
        k = np.random.choice([0, 1])
        strategy.append(k)
    
    # 随机生成 3 个 z 变量 (0 或 1)
    for i in range(1, 3):
        z = np.random.choice([0, 1])
        strategy.append(z)
    
    return strategy

# 测试生成随机策略
random_strategy = generate_random_strategy()
print("随机生成的策略组合: ", random_strategy)
# ======================================================================================
def assign_values(params):
    # 解包参数数组
    (x1, x2, x3, x4, x5, x6, x7, x8, 
     y1, y2, y3, 
     k1, k2, k3, 
     z1, z2) = params
    
    # 输出各变量值，用于确认赋值是否正确
    print(f"x1: {x1}, x2: {x2}, x3: {x3}, x4: {x4}, x5: {x5}, x6: {x6}, x7: {x7}, x8: {x8}")
    print(f"y1: {y1}, y2: {y2}, y3: {y3}")
    print(f"k1: {k1}, k2: {k2}, k3: {k3}")
    print(f"z1: {z1}, z2: {z2}")

# 示例使用，传入一个按顺序包含所有参数值的数组
values = [0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0,1]
print(len(values))
# 调用函数
assign_values(random_strategy)