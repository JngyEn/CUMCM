from math import ceil
import numpy as np
import scipy.stats as stats

class Q_1_result:
    def __init__(self,m,n,result,is_reach_max) -> None:
        # 当前已采样次数
        self.m=m
        # 已采样结果不合格次数
        self.n=n
        # result表示是否大于0.1或小于0.1, is_reach_max表示是否触及最大采样次数
        self.result= result
        self.is_reach_max=is_reach_max

def Q_1(test_array:list[bool],alpha:float,E:float,alternative:str)->Q_1_result:
    assert alternative == 'greater' or alternative == 'less'
    assert 1 > alpha > 0
    assert 1 > E > 0
    # test_array 有序的抽样结果,True表示为次品,False为良品
    # alpha 置信度
    # E 允许的误差
    # alternative表示检测大于0.1或小于0.1(greater/less)
    # 返回值result表示是否大于0.1或小于0.1, is_reach_max表示是否触及最大采样次数

    # 题目中出现的次品率p所需低于0.1
    P = 0.1 

    # 当前置信水平alpha的Z值
    Z = stats.norm.ppf(1-(1-alpha)/2)

    # 最大采样次数
    N = ceil((Z/E)**2*P*(1-P))

    # 当前已采样次数
    m = 0
    # 已采样结果不合格次数
    n = 0

    for i in range(min(len(test_array),N)):
        if test_array[i]:
            n += 1
        m += 1

        # 获取p值
        if alternative=="less":
            p_value = 1 - stats.binom.cdf(n, m, p=P)
            p_v2 = stats.binom.cdf(n-1, m, p=P)
        else:#greater
            p_value = stats.binom.cdf(n-1, m, p=P)
            p_v2 = 1 - stats.binom.cdf(n, m, p=P)
        #假设检验
        if p_value > alpha:
            return Q_1_result(m,n,True,False)
        elif p_v2 > alpha:
            return Q_1_result(m,n,False,False)
    #未返回说明达到最大采样次数,没有充足证据说明次品率大于或小于0.1
    return Q_1_result(m,n,False,True)

# 设置随机种子以便复现结果
np.random.seed(114514)

# 生成长度为array_length的随机数组，每个元素有prob_true的概率为True，1-prob_true的概率为False, 表示实际次品率为prob_true
array_length = 800
prob_true = 0.13
array = np.random.rand(array_length) < prob_true

result = Q_1(array,alpha=0.90,E=0.02,alternative='greater')
print(f"m={result.m},n={result.n},result={result.result},is_reach_max={result.is_reach_max}")

