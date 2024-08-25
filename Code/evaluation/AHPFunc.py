import numpy as np

def AHP(np_arr):
    print("===============AHP启动！=======================")
    # Step 1.构建判断矩阵
    #A = np.array([[1, 2, 3, 5], [1/2, 1, 1/2, 2], [1/3, 2, 1, 2], [1/5, 1/2, 1/2, 1]])
    A = np_arr
    # 获取指标数，便于从表中查值
    n = A.shape[0]


    # 1. 求最大特征值
    #eig_vec特征向量, eig_val 特征值
    eig_val,eig_vec = np.linalg.eig(A)

    # 特征值的最大值
    Max_eig = max(eig_val)

    # 2. 计算CI 并查询得RI，得到 CR
    CI = (Max_eig - n) / (n - 1) 
    RI = [0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59] 
    CR = CI/RI[n]

    print('CI=', CI)
    print('RI=', CR)

    if CR < 0.10:
        print('CR<0.10, correct! ')
    else:
        print('CR >= 0.10, adjust matrix')

        

    # 计算每列的和
    ASum = np.sum(A, axis=0)
    print('ASum=', ASum)
    # 归一化后的矩阵
    Stand_A = A / ASum
    Stand_A = np.round(Stand_A, 3)
    print(Stand_A)




    # 找出最大特征值的索引和对应的特征向量
    max_index = np.argmax(eig_val)
    max_vector = eig_vec[:,max_index]

    #  对特征向量进行归一化处理得到权重
    weights = max_vector / np.sum(max_vector)
    weights_abs = abs(weights) 
    weights_abs = np.round(weights_abs, 3)
    print(weights_abs)


    marks = [0] * n
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += Stand_A[i][j] * weights_abs[j]
        marks[i] =sum

    print("weighs are: ",marks)
    print("===============AHP结束=======================")
    return marks

