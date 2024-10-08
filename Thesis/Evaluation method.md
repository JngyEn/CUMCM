# 数据预处理
属性具有多种类型.包括效益型、成本型和区间型等。这三种属性，效益型属性越大 越好，成本型属性越小越好，区间型属性是在某个区间最佳
在进行决策时，一般要进行属性值的规范化，主要有如下三个作用:
1. 属性值有多种类型，上述三种属性放在同一个表中不便于直接从数值大小判断方案的优劣，因此需要对 数据进行预处理, 必须在综合评价之前将属性的类型作一致化处理，使得表中任一周性下性能越优的方案变换后的属性值越大。
2. 无量纲化. 多属性决策与评估的困难之一是属性间的不可公度性，即在属性值表中的每一列数值具有不同的单位（量纲）。即使对同一 属性，采用不同的计量单位，表中的数值也就不同。在用各种多属性决策方法进行分析评价时，需要排除量纲的选用对决策或评估结果的影响，这就是无量纲化。
3. 归一化. 属性值表中不同指标的属性值的数值大小差别很大，为了直观，更为了便于采用各种多属性决 策与评估方法进行评价，需要把属性值表中的数值归一化，即把表中数值均变换到\[0,1]区间上
此外，还可在属性规范时用非线性变换或其他办法，来解决或部分解决某些目标的达到程与属性值之间的非线性关系，以及目标间的不完全补偿性。
常用的属性规范化方法有以下几种
1. 线性变换.效益型属性:
$$
b_{ij} = a_{ij}/a_j^{max}
$$
	不再赘述成本型属性. 采用这种方式进行属性规范化, 效益型属性中经过变换的最差属性值不一定为0，最优属性值为1; 成本型属性中, 经过变换的最优属性值不一定为1，最差属性值为0。
2. 标准0-1变换. 为了使每个属性变换后的最优值为1 且最差值为0，令效益型属性
$$
b_{ij}={a_{ij}-a_j^{min} \over a_j^{max}-a_j^{min}}
$$
	不再赘述成本型属性
3. 区间型属性的变换。有些属性既非效益型又非成本型，如生师比。显然这种属性不能采用前面介绍的两种方法处理。设给定的最优属性区间为$[a_j^0,a_j^*]$，$a_j'$为无法容忍下限，$a_j^{''}$为无法容忍上限，则
$$
b_{ij}=\begin{cases}
1-(a_j^0-a_{ij})/(a_j^0-a_j'),\ \ \ a_j' \le a_{ij} < a_j^0\\
1,\ a_j^0 \le a_{ij} \le a_j^*\\
1-(a_{ij}-a_j^*)/(a_j^{''}-a_j^*),\ \ a_j^* \le a_{ij} \le a_j^{''}  \\ 
0,otherwise
\end{cases}
$$
4. 向量规范化. 无论成本型属性还是效益型属性，向量规范化均用下式进行变换∶

$$
b_{ij} = {a_{ij} \over \sqrt{\sum_{i=1}^m a_{ij}^2} }
$$
	它与前面介绍的几种变换不同，从变换后属性值的大小上无法分辨属性值的优劣。它的 最大特点是，规范化后，各方案的同一属性值的平方和为1.因此常用于计算各方案与某 种虚拟方案（如理想点或负理想点）的欧几里得距离的场合
5. 标准化处理. 在实际问题中，不同变量的测量单位往往是不一样的。为了消除变量的量纲效应，使每个变量都具有同等的表现力，数据分析中常对数据进行标准化处理，即
$$
\mu_j={1 \over m}\sum_{i=1}^m a_{ij}
$$
$$
s_j=\sqrt {{1 \over m-1}\sum_{i-1}^m(a_{ij}-\mu_j)^2}
$$
$$
b_{ij}={a_{ij}-\mu_j \over s_j}
$$
# 赋权
## 客观赋权
### Topsis 法(理想解法)
*这种方法通过构造评价问题的正理想解和负理想解、即各指标的最优解和最劣解，通过计算每个方案到理想方案的相对贴近度(距离), 来对方案进行排序*
#### 算法步骤
1. 用向量规范法的方法求得规范决策矩阵. 设多属性决策问题的决策矩阵$A=(a_{ij})_{m\times n}$, 规范化决策矩阵$B=(b_{ij})_{m\times n}$, 其中
$$
b_{ij} = {a_{ij} \over \sqrt{\sum_{i=1}^m a_{ij}^2} }
$$
2. 构造$C=(c_{ij})_{m\times n}$. 设由决策人给定各属性的权重向量为$w=[w_1,w_2,\ldots,w_n]^T$, 则
$$
c_{ij}=w_i*b_{ij}
$$
3. 确定正理想解与负理想解
$$
正理想解: c_j^*=\begin{cases}
maxc_{ij},j为效益型属性 \\
minc_{ij},j为成本型属性
\end{cases}
$$
	负理想解$c_j^0$不再赘述
4. 计算各方案到正理想解与负理想解的距离. 备选方案$d_i$到正理想解的距离为
$$
s_i^*=\sqrt {\sum_{j=1}^n (c_{ij}-c_j^*)^2}
$$
	备选方案$d_i$到负理想解的距离$s_i^0$不再赘述
5. 计算各方案的排序指标值（即综合评价指数）。即
$$
f^*_i= {s_i^0 \over (s_i^0+s_i^*)}
$$
6. 按由$f_i^*$大到小排列方案的优劣次序
### 熵权法
熵权法根据各指标的变异程度，利用信息熵计算出各指标的熵权，从而得出较为客观的指标权重
#### 评价步骤
设有$n$个评价对象，$m$个评价指标变量，第$i$个评价对象关于第$j$个指标变量的取值为$a_{ij}$，构造数据矩阵$A=(a_{ij})_{n\times m}$
1. 利用原始数据矩阵$A=(a_{ij})_{n\times m}$计算$p_{ij}$，即第$i$个评价对象关于第$j$个指标值的比重
$$
p_{ij}={a_{ij}\over\sum_{i=1}^n a_{ij}}
$$
2. 计算第$j$项指标的熵值
$$
e_j=-{1\over \ln n}\sum_{i=1}^np_{ij}\ln p_{ij}
$$
3. 计算第$j$项指标的变异系数
$$
g_j=1-e_j
$$
4. 计算第$j$项指标的权重
$$
w_j={g_j \over \sum_{j=1}^m g_j}
$$
5. 计算第$i$个评价对象的综合评价值
评价值越大越好
### 秩和比综合评价法
秩和比综合评价法基本原理是在一个$n$行$m$列矩阵中通过秩转换、获得无量纲统计量 RSR, 以 RSR 值对评价对象的优劣直接排序或分档排序，从而对评价对象做出综合评价。
#### 步骤
1. 编秩. 对于每个指标，将所有评价对象的数值进行排序，并为其分配秩次（即排名）。秩次的赋值方法可以是从最优到最差赋予 1 到 $m$ 的顺序
2. 计算秩和比（$RSR$）
$$
RSR_i={1 \over mn}\sum_{j=1}^m R_{ij}
$$
	这里也能赋权
3. 分档排序

## 主观赋权
*考的少, 出了说明考题low*
### 层次分析法
*好像比较low?*
#### 建立步骤
1. 建立层次结构模型
- **目标层**：最高层，表示最终要达成的决策目标。
- **准则层**：中间层，表示为达成目标所需要考虑的准则或标准。对于复杂问题，准则层可以进一步分为子准则层。
- **方案层**：最底层，表示具体的决策方案或备选方案。
	*此处应有图片*
2. 构造判断矩阵. - 对于每一层次中的每个准则，决策者需要成对比较其对上一层次要素的相对重要性。通过专家打分或经验估计，构建一个两两比较的判断矩阵。判断矩阵是一个对称矩阵，其中元素$a_{ij}$ 表示准则$i$相对于准则$j$的重要性. 例如1-9 标度法:
	- 1 表示两者同等重要
	- 3 表示前者稍微重要于后者
	- 5 表示前者明显重要于后者
	- 7 表示前者强烈重要于后者
	- 9 表示前者极端重要于后者
	倒数则相反
3. 计算权重向量. 参考[[#数据预处理]]求出特征向量并归一化
4. 一致性检验. 判断矩阵的构造可能会带来一定的不一致性，因此需要进行一致性检验。
	计算一致性比率（Consistency Ratio, $CR$）：
$$
CI= {\lambda_{max}-n \overn-1}
$$
	$\lambda_{max}$是判断矩阵的最大特征值，$n$是矩阵的阶数
	计算一致性比率 $CR$：
	$$
	CR={CI \over RI}
	$$
	其中，$RI$是随机一致性指数，其值根据矩阵阶数不同而定
	当 $CR<0.1$时，判断矩阵具有较好的一致性，否则需要调整判断矩阵
5. 将各层次的权重向量按照层次结构进行综合，计算出各备选方案的总排序权重。最终，排序权重最大的方案即为最优选择
# 不涉及赋权的评价方法

### 模糊综合评价
#### 一级模糊综合评判模型的建立步骤
1. 确定因素集(指标体系), 记为
$$
U=\{ u_1,u_2,\ldots,u_n \}
$$
2. 确定评语集(评价等级), 记为
$$
V=\{ v_1,v_2,\ldots,u_n \}
$$
3. 确定各因素权重, 记为
$$
A=[a_1,a_2,\ldots,a_n]
$$
	$a_i$为第i个因素的权重, 且满足$\sum_{i=1}^n a_i=1$
	确定权重的方法很多，如 Delphi法、加权平均法、众人评估法等。
4. 构建模糊综合判断矩阵。对指标$u_i$来说，对各个评语的隶属度为$V$上的模糊子 集。对指标$u_i$的评判记为
$$
R_i=[r_{i1},r_{i2},\dots,r_{im}]
$$
	各指标的模糊综合判断矩阵为
$$
R= \left[\begin{matrix}
r_{11}&r_{12}&\dots&r_{1m}\\
r_{21}&r_{22}&\dots&r_{2m}\\
\vdots&\vdots&\ddots&r_{1m}\\
r_{n1}&r_{n2}&\dots&r_{nm}
\end{matrix}
\right]
$$
	它是一个从$U$到$V$的模糊关系矩阵。
5. 进行模糊综合评判. **(这里教材疑似有问题)**  将权重向量$A$与模糊评价矩阵$R$进行综合，即通过加权平均法或其他合成方法计算得到综合评价向量$B$:
$$
B=A R=[b_1,b_2,\dots,b_n]
$$
6. 根据综合评价向量$B$，结合评价等级，确定最终的评价结果。常用的方法包括最大隶属度原则，即选择隶属度最大的等级作为最终评价结果；也可以根据隶属度的分布情况作进一步分析
#### 多层次模糊综合评价
对于—些复杂的系统，如*人事考核*中涉及的指标较多时，需要考虑的因素很多. 这时,如果仍用一级模糊综合评判，则会出现两个方面的问题: 一是因素过多。它们的**权数分配难以确定**; 另一方面，即使确定了权分配，由于需要满足归一化条件, **每个因素的权值都小**，对这种系统，可以采用多层次模糊综合评判方法。对于人事考核而言，采用二级系统 就足以解决问题了，如果实际中要划分更多的层次，那么可以用二级模糊综合评判的方法类推

### 灰色关联分析
#### 评价步骤
1. 确定参考序列与比较序列. 参考序列通常为理想序列或目标序列，是分析的标准, 一般取最优方案$a_0(j)=\max \{ a_i(j) \}$，也可以是期望值。比较序列：为需要分析的各个方案的实际指标序列
2. 数据预处理. 参考[[#数据预处理]] ,并且此时可以通过[[#赋权|其他方法赋权]]确定相应权重
3. 计算灰色关联度系数
$$
\zeta_j(k)={\min_{j,k}\Delta_j(k)+\rho*\max_{j,k}\Delta_j(k)
\over \Delta_j(k) + \rho*\max_{j,k}\Delta_j(k)
}
$$
	其中:
	- $\Delta_j(k)=|X_0(k)-X_j(k)|$ 是参考序列与比较序列在第$k$个指标上的差值
	- $ρ$ 是分辨系数，取值范围在 $0≤ρ≤1$ 之间，通常取 $0.5$. 通过调整$ρ$的大小，可以改变关联系数对差异的敏感度。较大的 $ρ$ 值会使得关联系数对较大差异更敏感，较小的 $ρ$ 值则会使得关联系数更均衡，不会过度强调某些极端值
4. 计算关联度. 第$j$个方案的关联度 $r_j​$ 通常为关联系数的均值(灰色加权关联)
$$
r_j={1 \over n}\sum_{k=1}^n\zeta_j(k)
$$
5. 评价分析。根据灰色加权关联度的大小，对各评价对象进行排序，可建立评价对象的关联序，关联度越大，其评价结果越好。
## 神经网络
*不会啊,你们会吗*