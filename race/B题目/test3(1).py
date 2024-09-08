from scipy.stats import beta

m=800
n=73

# Beta 分布参数
alpha = 1+m-n
beta_param = 1+n

# 计算 P(p > 0.9)
p_greater_than_0_9 = 1 - beta.cdf(0.9, alpha, beta_param)

print(f"P(p > 0.9 | X = {m-n}) = {p_greater_than_0_9:.4f}")