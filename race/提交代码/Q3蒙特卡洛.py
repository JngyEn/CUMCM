import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# ======================================================================================数据处理
dfComponent = pd.read_excel("C:/Users/JngyEn/Downloads/2024/B题/table2/2零件.xlsx")
dfHalfProduct = pd.read_excel("C:/Users/JngyEn/Downloads/2024/B题/table2/2半成品.xlsx")
dfProduct = pd.read_excel("C:/Users/JngyEn/Downloads/2024/B题/table2/2成品.xlsx")

dfComponent.columns = dfComponent.columns.str.replace(' ', '', regex=True)
dfHalfProduct.columns = dfHalfProduct.columns.str.replace(' ', '', regex=True)
dfProduct.columns = dfProduct.columns.str.replace(' ', '', regex=True)


dfComponent.head(10)
dfHalfProduct.head()
dfProduct.head()

# 遍历每个零配件，半成品，成品并赋值给变量

# 零配件数据
for index, row in dfComponent.iterrows():
    globals()[f'component_{index+1}_defect_rate'] = row['次品率']
    globals()[f'single_component_{index+1}_purchase_price'] = row['购买单价']
    globals()[f'single_component_{index+1}_inspection_cost'] = row['检测成本']

# 半成品数据
for index, row in dfHalfProduct.iterrows():
    globals()[f'halfProduct_{index+1}_defect_rate'] = row['半成品次品率']
    globals()[f'single_halfProduct_{index+1}_assembly_cost'] = row['半成品装配成本']
    globals()[f'single_halfProduct_{index+1}_inspection_cost'] = row['半成品检测成本']
    globals()[f'single_halfProduct_{index+1}_dismantling_cost'] = row['半成品拆解费用']

# 成品数据
for index, row in dfProduct.iterrows():
    globals()[f'product_defect_rate'] = row['成品次品率']
    globals()[f'single_product_assembly_cost'] = row['成品装配成本']
    globals()[f'single_product_inspection_cost'] = row['成品检测成本']
    globals()[f'single_product_dismantling_cost'] = row['成品拆解费用']
    globals()[f'single_product_market_price'] = row['成品市场售价']
    globals()[f'single_product_exchange_loss'] = row['成品调换损失']
# ======================================================================================蒙特卡洛随机生成


# 定义一个函数来生成 xy 和 kz 变量的随机组合
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

# ======================================================================================利润函数

def total_profie(params):

    # 解包参数数组
    (x1, x2, x3, x4, x5, x6, x7, x8, 
     y1, y2, y3, 
     k1, k2, k3, 
     z1, z2) = params
    
    # 将解包的变量全部放入 globals()
    globals()['x1'] = x1
    globals()['x2'] = x2
    globals()['x3'] = x3
    globals()['x4'] = x4
    globals()['x5'] = x5
    globals()['x6'] = x6
    globals()['x7'] = x7
    globals()['x8'] = x8

    globals()['y1'] = y1
    globals()['y2'] = y2
    globals()['y3'] = y3

    globals()['k1'] = k1
    globals()['k2'] = k2
    globals()['k3'] = k3

    globals()['z1'] = z1
    globals()['z2'] = z2

    
    


    # ======================================================================================检测决策后的次品率
    # 半成品检测后良品率

    globals()[f'halfProduction_1_afterInspection_good_rate'] = (1 - globals()[f'halfProduct_1_defect_rate'] * ( 1 - globals()[f'y1']))    *    (1 - globals()[f'component_1_defect_rate'] * ( 1 - globals()[f'x1']))    *    (1 - globals()[f'component_2_defect_rate'] * ( 1 - globals()[f'x2']))   *    (1 - globals()[f'component_3_defect_rate'] * ( 1 - globals()[f'x3']))

    globals()[f'halfProduction_2_afterInspection_good_rate'] = (1 - globals()[f'halfProduct_2_defect_rate'] * ( 1 - globals()[f'y2']))    *    (1 - globals()[f'component_4_defect_rate'] * ( 1 - globals()[f'x4']))    *    (1 - globals()[f'component_5_defect_rate'] * ( 1 - globals()[f'x5']))   *    (1 - globals()[f'component_6_defect_rate'] * ( 1 - globals()[f'x6']))

    globals()[f'halfProduction_3_afterInspection_good_rate'] = (1 - globals()[f'halfProduct_2_defect_rate'] * ( 1 - globals()[f'y3']))    *    (1 - globals()[f'component_7_defect_rate'] * ( 1 - globals()[f'x7']))    *    (1 - globals()[f'component_8_defect_rate'] * ( 1 - globals()[f'x8']))   

    # 半成品检测后次品率

    halfProduct_1_afterInspection_defect_rate = 1 - globals()[f'halfProduction_1_afterInspection_good_rate']
    halfProduct_2_afterInspection_defect_rate = 1 - globals()[f'halfProduction_2_afterInspection_good_rate']
    halfProduct_3_afterInspection_defect_rate = 1 - globals()[f'halfProduction_3_afterInspection_good_rate']
    # 成品检验决策后的良品率
    product_afterInspection_good_rate = (1 - globals()[f'product_defect_rate'] * ( 1 - globals()[f'z1']))    *    (1 - halfProduct_1_afterInspection_defect_rate * ( 1 - globals()[f'y1']))    *    (1 - halfProduct_2_afterInspection_defect_rate * ( 1 - globals()[f'y2']))   *    (1 - halfProduct_3_afterInspection_defect_rate * ( 1 - globals()[f'y3']))

    # 成品检验决策后的次品率
    product_afterInspection_defect_rate = 1 - product_afterInspection_good_rate

    # ======================================================================================购买数量
    # 生产成品数，假设
    Total_production_num = 10000
    # 半成品数量
    halfProduct_1_num = Total_production_num / (1- halfProduct_1_afterInspection_defect_rate * (1 - y1))
    halfProduct_2_num = Total_production_num / (1- halfProduct_2_afterInspection_defect_rate * (1 - y2))
    halfProduct_3_num = Total_production_num / (1- halfProduct_3_afterInspection_defect_rate * (1 - y3))


    # 零件购买数量
    for index in range(1, 4):
        # 动态生成变量名并进行计算
        globals()[f'total_component_{index}_purchase_num'] = halfProduct_1_num / (1 - globals()[f'component_{index}_defect_rate'] * (1 - globals()[f'x{index}']))

    # 零件数量
    for index in range(4, 7):
        # 动态生成变量名并进行计算
        globals()[f'total_component_{index}_purchase_num'] = halfProduct_2_num / (1 - globals()[f'component_{index}_defect_rate'] * (1 - globals()[f'x{index}']))
        
    # 零件数量
    for index in range(7, 9):
        # 动态生成变量名并进行计算
        globals()[f'total_component_{index}_purchase_num'] = halfProduct_3_num / (1 - globals()[f'component_{index}_defect_rate'] * (1 - globals()[f'x{index}']))

    # ======================================================================================购买价格
    # 所有的零件购买价格
    total_component_cost = 0
    for index in range(1, 9):
        globals()[f'total_component_{index}_purchase_cost'] = globals()[f'total_component_{index}_purchase_num'] * globals()[f'single_component_{index}_purchase_price']
        total_component_cost += globals()[f'total_component_{index}_purchase_cost']



    # ======================================================================================检测成本
    # 所有的零件检测成本
    total_component_inspection_cost = 0
    for index in range(1, 4):
        # 动态生成变量名并进行计算
        globals()[f'total_component_{index}_inspection_cost'] = globals()[f'x{index}'] * halfProduct_1_num / (1 - globals()[f'component_{index}_defect_rate'] ) * globals()[f'single_component_{index}_inspection_cost']
        total_component_inspection_cost += globals()[f'total_component_{index}_inspection_cost']


    for index in range(4, 7):
        # 动态生成变量名并进行计算
        globals()[f'total_component_{index}_inspection_cost'] = globals()[f'x{index}'] * halfProduct_2_num / (1 - globals()[f'component_{index}_defect_rate'] ) * globals()[f'single_component_{index}_inspection_cost']
        total_component_inspection_cost += globals()[f'total_component_{index}_inspection_cost']

    for index in range(7, 9):
        # 动态生成变量名并进行计算
        globals()[f'total_component_{index}_inspection_cost'] = globals()[f'x{index}'] * halfProduct_3_num / (1 - globals()[f'component_{index}_defect_rate'] ) * globals()[f'single_component_{index}_inspection_cost']
        total_component_inspection_cost += globals()[f'total_component_{index}_inspection_cost']

    
   

    total_halfProduct_inspection_cost = 0

    # 所有半成品的检测成本
    total_halfProduct_1_inspection_cost = y1 * Total_production_num / (1 - halfProduct_1_afterInspection_defect_rate ) * globals()[f'single_halfProduct_1_inspection_cost']
    total_halfProduct_2_inspection_cost = y2 * Total_production_num / (1 - halfProduct_2_afterInspection_defect_rate ) * globals()[f'single_halfProduct_2_inspection_cost']
    total_halfProduct_3_inspection_cost = y3 * Total_production_num / (1 - halfProduct_3_afterInspection_defect_rate ) * globals()[f'single_halfProduct_3_inspection_cost']

    total_halfProduct_inspection_cost = total_halfProduct_1_inspection_cost + total_halfProduct_2_inspection_cost + total_halfProduct_3_inspection_cost


    # 成品的检测成本
    total_product_inspection_cost = z1 * Total_production_num * globals()[f'single_product_inspection_cost']    

    # ======================================================================================

    total_halfProduct_assembly_cost = 0
    # 半成品的装配成本
    for index in range(1, 4):
        # 动态生成变量名并进行计算
        globals()[f'total_halfProduct_{index}_assembly_cost'] = halfProduct_1_num * globals()[f'single_halfProduct_{index}_assembly_cost']
        total_halfProduct_assembly_cost += globals()[f'total_halfProduct_{index}_assembly_cost']
        
  
    # 成品的装配成本

    total_product_assembly_cost = Total_production_num * globals()[f'single_product_assembly_cost'] 

    # ======================================================================================

    # 成品的拆解个数,同时也是调换个数
    total_product_dismantling_num = z2 * Total_production_num * product_afterInspection_defect_rate
    # 成品的拆解成本 
    total_product_dismantling_cost =(total_product_dismantling_num  * 
                                        (globals()[f'single_product_dismantling_cost'] + globals()[f'single_product_inspection_cost'] +
                                        (1-k1)*(1 - y1)* globals()[f'single_halfProduct_1_inspection_cost'] + (1-k2)*(1 - y2)* globals()[f'single_halfProduct_2_inspection_cost'] +  (1-k3)*(1 - y3)* globals()[f'single_halfProduct_3_inspection_cost']
                                        ))       
                                             
    # 半成品的拆解成本
    total_halfProduct_dismantling_cost = 0
    globals()[f'total_halfProduct_1_dismantling_cost'] =( k1 * total_product_dismantling_num * (globals()[f'single_halfProduct_1_dismantling_cost'] - 
                                                                                                (globals()[f'single_component_1_purchase_price'] + globals()[f'single_component_2_purchase_price'] + globals()[f'single_component_3_purchase_price']) +
                                                                                                    ((1-x1)*globals()[f'single_component_1_inspection_cost']+(1-x2)*globals()[f'single_component_2_inspection_cost'])+(1-x3)*globals()[f'single_component_3_inspection_cost']
                                                                                                    ))

    globals()[f'total_halfProduct_2_dismantling_cost'] =( k2 * total_product_dismantling_num * (globals()[f'single_halfProduct_2_dismantling_cost'] - 
                                                                                                (globals()[f'single_component_6_purchase_price'] + globals()[f'single_component_4_purchase_price'] + globals()[f'single_component_5_purchase_price']) +
                                                                                                    ((1-x6)*globals()[f'single_component_6_inspection_cost']+(1-x4)*globals()[f'single_component_4_inspection_cost'])+(1-x5)*globals()[f'single_component_5_inspection_cost']
                                                                                                    ))

    globals()[f'total_halfProduct_3_dismantling_cost'] =( k3 * total_product_dismantling_num * (globals()[f'single_halfProduct_3_dismantling_cost'] - 
                                                                                                (globals()[f'single_component_7_purchase_price'] + globals()[f'single_component_8_purchase_price'])  +
                                                                                                    ((1-x7)*globals()[f'single_component_7_inspection_cost']+(1-x8)*globals()[f'single_component_8_inspection_cost'])
                                                                                                    ))
    # 求总拆解成本
    total_halfProduct_dismantling_cost = (globals()[f'total_halfProduct_1_dismantling_cost'] +
                                        globals()[f'total_halfProduct_2_dismantling_cost'] +
                                        globals()[f'total_halfProduct_3_dismantling_cost'])
  

    # ======================================================================================
    # 成品的调换损失

    total_product_exchange_loss = total_product_dismantling_num * globals()[f'single_product_exchange_loss']
    # ======================================================================================
    Total_cost = (total_component_cost +
                (total_component_inspection_cost + total_halfProduct_inspection_cost + total_product_inspection_cost) +
                (total_halfProduct_assembly_cost + total_product_assembly_cost)+
                (total_halfProduct_dismantling_cost + total_product_dismantling_cost)+
                total_product_exchange_loss)
   

    # ======================================================================================

    # 总收入
    Total_income = Total_production_num * globals()[f'single_product_market_price']

  

    # 总利润
    Total_profit = Total_income - Total_cost

 
    return(Total_profit)


# ======================================================================================蒙卡特罗
# 设定模拟次数
iterations = 10000
# 初始化存储每次迭代的组合和对应的利润值
strategies = []
profits = []
# 初始化变量以存储最佳策略和最大利润
best_strategy = None
max_profit = -np.inf

for i in range(iterations):
    random_strategy = generate_random_strategy()
    strategies.append(random_strategy)
    Total_profit = total_profie(random_strategy)
    if Total_profit > max_profit:
        max_profit = Total_profit
        best_strategy = random_strategy  # 这里存储最优策略，应该包括所有的决策变量

    profits.append(Total_profit)

# 将组合类型转换为字符串，用于绘图的横坐标
strategy_strings = [''.join(map(str, strategy)) for strategy in strategies]
# 绘制利润值和策略组合的图
# 绘制利润值和策略组合的图
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 使用白色填充和蓝色边框的样式绘制点图
plt.scatter(strategy_strings, profits, facecolors='white', edgecolors='blue')

plt.xlabel('决策组合类型')
plt.ylabel('利润')
plt.title('决策组合类型与利润的关系')  # 调整标签角度以适应较长的组合名称
# plt.tight_layout()  # 自动调整布局
plt.show()
# 输出最优结果

print(f"最佳策略: {best_strategy}")
# 输出策略中所有的 x, y, k, z 参数值
# 输出 8 个 x 变量
for i in range(8):
    print(f"x{i+1} = {best_strategy[i]}")

# 输出 3 个 y 变量
for i in range(8, 11):
    print(f"y{i-7} = {best_strategy[i]}")

# 输出 3 个 k 变量
for i in range(11, 14):
    print(f"k{i-10} = {best_strategy[i]}")

# 输出 2 个 z 变量
for i in range(14, 16):
    print(f"z{i-13} = {best_strategy[i]}")
print(f"最大利润: {max_profit}")
for i in range(iterations):
    random_strategy = generate_random_strategy()
    Total_profit = total_profie(random_strategy)
    profits.append(Total_profit)  # 将每次的利润存储到列表中
    

# 计算利润的期望和标准差
mean_profit = np.mean(profits)
std_profit = np.std(profits)

# 输出结果
print(f"在 {iterations} 次模拟中的期望利润: {mean_profit}")
print(f"在 {iterations} 次模拟中的利润标准差: {std_profit}")