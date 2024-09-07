import pandas as pd
import matplotlib.pyplot as plt
import itertools

# 读取Excel文件
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义一个函数，根据不同的情况选择对应的参数，返回利润、成本和成品率
def get_profit_cost_yield(case_number, x1, x2, x3, x4, N):
    row = df.iloc[case_number - 1]
    # 零配件1的次品率、检测成本、购买单价
    a1 = row['零配件1次品率']
    a2 = row['零配件1检测成本']
    a3 = row['零配件1购买单价']
    
    # 零配件2的次品率、检测成本、购买单价
    b1 = row['零配件2次品率']
    b2 = row['零配件2检测成本']
    b3 = row['零配件2购买单价']
    
    # 成品的次品率、检测成本、装配成本、市场售价
    c1 = row['成品次品率']
    c2 = row['成品检测成本']
    c3 = row['成品装配成本']
    c4 = row['成品市场售价']
    
    # 不合格成品的调换损失和拆解费用
    d1 = row['调换费用']
    d2 = row['拆解费用']
    
    # 组装后的次品的概率
    c_good = (1 - a1 * (1 - x1)) * (1 - b1 * (1 - x2)) * (1 - c1 * (1 - x3))
    c5 = 1 - c_good
    c_defective_num = c5 * N
    c_sell_num = N - c_defective_num

    # 采购成本和检测成本
    a_N = N / (1 - a1 * (1 - x1))
    a_cost = a3 * a_N
    a_detect_cost = x1 * a2 * N / (1 - a1)
    
    b_N = N / (1 - b1 * (1 - x2))
    b_cost = b3 * b_N
    b_detect_cost = b2 * N / (1 - b1) * x2
    
    # 成品装配和检测成本
    c_cost = N * c3
    c_detect_cost = N * c2 * x3
    
    # 调换损失
    c_lose = (1 - x3) * c_defective_num * d1

    # 拆解后零件回收成本
    c_dismantle_cost = x4 * ((d2 + c3) - (a3 + b3) + ((1 - x1) * a2 + (1 - x2) * b2 + (1 - x3) * c2)) * c_defective_num

    # 总成本
    Total_cost = a_cost + a_detect_cost + b_cost + b_detect_cost + c_cost + c_detect_cost + c_lose + c_dismantle_cost
    
    # 总利润
    Total_profit = c_sell_num * c4 - Total_cost
    
    # 返回利润、成本和成品率（良品率）
    return Total_profit, Total_cost, c_good

# 读取 Excel 表格
df = pd.read_excel("C:/Users/JngyEn/Downloads/2024/B题/表1test.xlsx")
df.columns = df.columns.str.replace(' ', '', regex=True)

# 假设总成品量
N = 100

# 生成所有参数组合
combinations = list(itertools.product([0, 1], repeat=4))

# 存储所有利润、成本和成品率结果
profits = {case: [] for case in range(1, 7)}
costs = {case: [] for case in range(1, 7)}
yields = {case: [] for case in range(1, 7)}

# 遍历每一种参数组合
for comb_index, (x1, x2, x3, x4) in enumerate(combinations):
    for case_number in range(1, 7):
        # 获取对应的利润、成本和成品率
        profit, cost, yield_rate = get_profit_cost_yield(case_number, x1, x2, x3, x4, N)
        profits[case_number].append(profit)      # 存储利润
        costs[case_number].append(cost)          # 存储成本
        yields[case_number].append(yield_rate)   # 存储成品率

# 绘制利润图像
plt.figure(figsize=(10, 6))
for case_number in range(1, 7):
    plt.plot(range(1, len(combinations) + 1), profits[case_number], label=f"情况 {case_number} 利润",
             marker='o', markerfacecolor='white')

plt.title('各参数组合下的利润情况')
plt.xlabel('参数组合编号')
plt.ylabel('利润')
plt.legend(title='情况')
plt.grid(axis='y')
plt.show()

# 绘制成本图像
plt.figure(figsize=(10, 6))
for case_number in range(1, 7):
    plt.plot(range(1, len(combinations) + 1), costs[case_number], label=f"情况 {case_number} 成本",
             marker='o', markerfacecolor='white')

plt.title('各参数组合下的成本情况')
plt.xlabel('参数组合编号')
plt.ylabel('成本')
plt.legend(title='情况')
plt.grid(axis='y')
plt.show()

# 绘制成品率图像
plt.figure(figsize=(10, 6))
for case_number in range(1, 7):
    plt.plot(range(1, len(combinations) + 1), yields[case_number], label=f"情况 {case_number} 成品率",
             marker='o', markerfacecolor='white')

plt.title('各参数组合下的成品率情况')
plt.xlabel('参数组合编号')
plt.ylabel('成品率')
plt.legend(title='情况')
plt.grid(axis='y')
plt.show()
