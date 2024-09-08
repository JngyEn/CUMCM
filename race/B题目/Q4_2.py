# 读取Excel文件
import itertools
from matplotlib import pyplot as plt
import scipy
import numpy as np
import pandas as pd
from scipy import stats
N=10000

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 定义一个函数，根据不同的情况选择对应的参数，返回利润
def get_profit(case_number, x1, x2, x3, x4, N):
    # x1 零配件1是否检测
    # x2 零配件2是否检测
    # x3 成品是否检测
    # x4 不合格成品是否拆解

    row = df.iloc[case_number - 1]
    # 零配件1的次品率、检测成本、购买单价

    a1 = defective_rate[case_number][0]   # 零配件1次品率

    a2 = row['零配件1检测成本']  # 零配件1检测成本
    a3 = row['零配件1购买单价']  # 零配件1购买单价

    # 零配件2的次品率、检测成本、购买单价

    b1 = defective_rate[case_number][1]   # 零配件2次品率

    b2 = row['零配件2检测成本']  # 零配件2检测成本
    b3 = row['零配件2购买单价']  # 零配件2购买单价

    # 成品的次品率、检测成本、装配成本、市场售价

    c1 = defective_rate[case_number][2]       # 成品次品率

    c2 = row['成品检测成本']     # 成品检测成本
    c3 = row['成品装配成本']     # 成品装配成本
    c4 = row['成品市场售价']     # 成品市场售价

    # 不合格成品的调换损失和拆解费用
    d1 = row['调换费用']         # 调换损失
    d2 = row['拆解费用']         # 拆解费用
 
    # 组装后的次品的概率
    # 最终良品率
    c_good = (1 - a1 * (1 - x1)) * (1 - b1 * (1 - x2)) * (1 - c1 * (1 - x3))
    # 最终次品率
    c5 = 1 - c_good

    # 最终次品个数
    c_defective_num = c5 * N 

    # 最终成品个数
    c_sell_num = N - c_defective_num

    # 采购数量
    a_N = N / (1-a1 * (1 - x1))

    # 采购成本
    a_cost = a3 * a_N

    # 检测成本
    a_detect_cost =  x1 * a2 * N / (1 - a1)
    # a_N*a2*x1

    # 采购数量
    b_N = N / (1-b1 * (1 - x2))

    # 采购成本
    b_cost = b3 * b_N

    # 检测成本
    b_detect_cost = b2 * N / (1 - b1) * x2

    # 成品装配和检测成本
    c_cost = N * c3
    # 检测成本
    c_detect_cost = N * c2 * x3

    # 调换损失和拆解成本
    c_lose = (1 - x3) * c_defective_num * d1

    # 回收成本与零件回收
    c_dismantle_cost = x4 * ((d2+c3) - (a3 + b3) + ((1 - x1) * a2 + (1 - x2)*b2 + (1-x3)*c2))*c_defective_num   

    Total_cost = a_cost + a_detect_cost + b_cost + b_detect_cost + c_cost + c_detect_cost +  c_lose + c_dismantle_cost

    Total_profit = c_sell_num * c4 - Total_cost

    return Total_profit

# 读取 Excel 表格
df = pd.read_excel("dataset/表1test.xlsx")
df.columns = df.columns.str.replace(' ', '', regex=True)

# 初始化Beta函数,最低检测次数
alpha = [[1]*3 for _ in range(7)] #1-index 所以是7
beta = [[1]*3 for _ in range(7)] 
min_detections = 100

# 生成长度为array_length的随机数组
np.random.seed(1919810)
array_length = 800
a_array = np.random.rand(array_length)
b_array = np.random.rand(array_length)
c_array = np.random.rand(array_length)

#开始试生产
for i in range(min(100,array_length)):
    #更新beta函数
    #6种情况
    for case_number in range(1,7):
        row = df.iloc[case_number - 1]
        a_defective_real_rate = row['零配件1次品率']
        b_defective_real_rate = row['零配件2次品率']
        c_defective_real_rate = row['成品次品率']
        
        if a_array[i] >= a_defective_real_rate:
            #零配件1不为次品,beta++
            beta[case_number][0]+=1
        else:
            #反过来alpha++
            alpha[case_number][0]+=1
        if b_array[i] >= b_defective_real_rate:
            #以下同理
            beta[case_number][1]+=1
        else:
            alpha[case_number][1]+=1
        if c_array[i] >= c_defective_real_rate:
            beta[case_number][2]+=1
        else:
            alpha[case_number][2]+=1

'''
选择方案
'''

# 生成所有参数组合
combinations = list(itertools.product([0, 1], repeat=4))

#确定次品率
defective_rate = [[0]*3 for _ in range(7)]
for case_number in range(1,7):
    for i in range(3):
        defective_rate[case_number][i] = stats.beta.mean(alpha[case_number][i], beta[case_number][i])

# 存储所有利润结果
profits = {case: [] for case in range(1, 7)}
# 存储每种情况下的最优组合
best_combination = {case: {"max_profit": -float('inf'), "combination": None} for case in range(1, 7)}

# 遍历每一种参数组合
for comb_index, (x1, x2, x3, x4) in enumerate(combinations):
    print(f"参数组合 {comb_index + 1}: x1={x1}, x2={x2}, x3={x3}, x4={x4}")
    
    # 对6种情况计算利润
    for case_number in range(1, 7):
        # 获取对应的情况
        profit = get_profit(case_number, x1, x2, x3, x4, N)
        profits[case_number].append(profit)

        # 判断当前利润是否是当前情况的最大利润
        if profit > best_combination[case_number]["max_profit"]:
            best_combination[case_number]["max_profit"] = profit
            best_combination[case_number]["combination"] = (x1, x2, x3, x4)

# 输出每种情况的最优方案
for case_number in range(1, 7):
    print(f"情况 {case_number} 的最优组合是 {best_combination[case_number]['combination']}，最大利润为 {best_combination[case_number]['max_profit']}")

changeflag=False
for i in range(100,array_length):
    for case_number in range(1,7):
        x1,x2,x3,x4 = best_combination[case_number]["combination"]
        #更新beta函数
        row = df.iloc[case_number - 1]
        a_defective_real_rate = row['零配件1次品率']
        b_defective_real_rate = row['零配件2次品率']
        c_defective_real_rate = row['成品次品率']
    
        if x1:
            if a_array[i] >= a_defective_real_rate:
                #a不为次品,beta++
                beta[case_number][0]+=1
            else:
                alpha[case_number][0]+=1
        
        if x2:
            if b_array[i] >= b_defective_real_rate:
                #b不为次品,beta++
                beta[case_number][1]+=1
            else:
                alpha[case_number][1]+=1
        
        #更新零配件1,2次品率
        defective_rate[case_number][0] = stats.beta.mean(alpha[case_number][0], beta[case_number][0])
        defective_rate[case_number][1] = stats.beta.mean(alpha[case_number][1], beta[case_number][1])

        #当x3&x2&x1时,可以更新 defective_rate[case_number][2]
        if x3:
            if x1 and x2:
                if c_array[i] >= c_defective_real_rate:
                    #成品不为次品,beta++
                    beta[case_number][2]+=1
                else:
                    #反过来alpha++
                    alpha[case_number][2]+=1

        # 存储每种情况下的最优组合
        the_best_combination = {case: {"max_profit": -float('inf'), "combination": None} for case in range(1, 7)}
        for comb_index, (x1, x2, x3, x4) in enumerate(combinations):
            # 获取对应的情况
            profit = get_profit(case_number, x1, x2, x3, x4, N)
            profits[case_number].append(profit)
            if profit > the_best_combination[case_number]["max_profit"]:
                the_best_combination[case_number]["max_profit"] = profit
                the_best_combination[case_number]["combination"] = (x1, x2, x3, x4)
        
        # 判断最优方案是否有变动
        if the_best_combination[case_number]["combination"] != best_combination[case_number]["combination"]:
            changeflag=True
            print("-----------------------------------------")
            print(f"在{i}次生产中最优方案发生改变")
            print(f"情况{case_number}最优方案已经由{best_combination[case_number]["combination"]}转为{the_best_combination[case_number]["combination"]}")
            t1,t2,t3,t4 = best_combination[case_number]["combination"]
            print(f"原有最优方案利润现在为{get_profit(case_number, t1, t2, t3, t4, N)}")
            t1,t2,t3,t4 = the_best_combination[case_number]["combination"]
            print(f"现有最优方案利润现在为{get_profit(case_number, t1, t2, t3, t4, N)}")
            print("-----------------------------------------")
            best_combination[case_number] = the_best_combination[case_number].copy()

if not changeflag:
    print("方案无改变")