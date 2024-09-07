import scipy.stats as stats

# 样本量和不合格数
m = 5000  # 样本总数
n = 450   # 不合格样品数
p0 = 0.9  # 假设的合格率

# 计算至少有 m-n 个合格样品时的概率（右尾累积概率）
# m - n 表示观测到的合格样品数
successes = m - n

# 使用survival function (1 - CDF)，计算 P(X >= successes) 当 p = 0.9 时
p_value = stats.binom.sf(successes - 1, m, p0)  # successes-1是因为 survival function 计算的是严格大于

# 打印结果
print(f"P(X >= {successes} | p = 0.9) = {p_value}")

# 判断是否拒绝零假设
alpha = 0.05  # 显著性水平
if p_value < alpha:
    print(f"置信度超过95%,拒绝零假设,合格率大于0.9")
else:
    print(f"无法拒绝零假设,合格率可能不超过0.9")

