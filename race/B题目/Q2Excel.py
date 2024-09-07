import pandas as pd
import matplotlib.pyplot as plt
import itertools

# 读取Excel文件
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义一个函数，根据不同的情况选择对应的参数，返回成本和利润
def get_profit(case_number, x1, x2, x3, x4, N):
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

    # 最终次品个数
    c_defective_num = c5 * N 

    # 最终成品个数
    c_sell_num = N - c_defective_num

    # 采购数量
    a_N = N / (1 - a1 * (1 - x1))
    a_cost = a3 * a_N
    a_detect_cost =  x1 * a2 * N / (1 - a1)

    b_N = N / (1 - b1 * (1 - x2))
    b_cost = b3 * b_N
    b_detect_cost = b2 * N / (1 - b1) * x2

    # 成品装配和检测成本
    c_cost = N * c3
    c_detect_cost = N * c2 * x3

    # 调换损失和拆解成本
    c_lose = (1 - x3) * c_defective_num * d1
    c_dismantle_cost = x4 * ((d2 + c3) - (a3 + b3) + ((1 - x1) * a2 + (1 - x2) * b2 + (1 - x3) * c2)) * c_defective_num   

    Total_cost = a_cost + a_detect_cost + b_cost + b_detect_cost + c_cost + c_detect_cost + c_lose + c_dismantle_cost
    Total_profit = c_sell_num * c4 - Total_cost

    return Total_cost, Total_profit

# 读取 Excel 表格
df = pd.read_excel("C:/Users/JngyEn/Downloads/2024/B题/表1test.xlsx")
df.columns = df.columns.str.replace(' ', '', regex=True)

# 假设总成品量
N = 10000

# 生成所有参数组合
combinations = list(itertools.product([0, 1], repeat=4))

# 创建表格来存储所有参数组合和6种情况的成本和利润
results_table = pd.DataFrame(columns=['组合编号', 'x1', 'x2', 'x3', 'x4'] + [f'情况{case}_成本' for case in range(1, 7)] + [f'情况{case}_利润' for case in range(1, 7)])

# 遍历每一种参数组合
for comb_index, (x1, x2, x3, x4) in enumerate(combinations):
    row_data = [comb_index + 1, x1, x2, x3, x4]  # 存储组合编号和参数
    for case_number in range(1, 7):
        total_cost, total_profit = get_profit(case_number, x1, x2, x3, x4, N)
        row_data.extend([total_cost, total_profit])
    
    # 添加到表格中
    results_table.loc[comb_index] = row_data

# 打印表格
import ace_tools as tools;
tools.display_dataframe_to_user(name="参数组合下的成本和利润", dataframe=results_table)
