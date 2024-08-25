### 最短路径
学过, 略
### 最小生成树
学过, 略
### 着色问题
着色算法目前还没有找到最优算法, (暴力嗯算)
#### python解决图着色问题
##### 1. **NetworkX**
`NetworkX` 是一个用于复杂网络分析的 Python 库，提供了图的创建、操作和可视化功能。它也包含一些图着色的功能，可以解决基本的图着色问题。
```python
import networkx as nx
import matplotlib.pyplot as plt
# 创建一个图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
# 使用 NetworkX 内置的图着色算法
coloring = nx.coloring.greedy_color(G, strategy="largest_first")
# 绘制图和着色
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=[coloring.get(node, 0) for node in G.nodes])
plt.show()
# 输出着色结果
print("节点着色结果:", coloring)
```
##### 2. `Google OR-Tools`
`Google OR-Tools` 是一个开源的运筹优化工具包，支持多种优化任务，包括图着色。OR-Tools 提供了强大的求解器，可以用于解决图着色问题。
```python
from ortools.sat.python import cp_model
# 创建模型
model = cp_model.CpModel()
# 定义变量
nodes = range(5)  # 假设有5个节点
colors = range(3)  # 假设使用3种颜色
# 为每个节点分配一个颜色
color_vars = {node: model.NewIntVar(0, len(colors) - 1, f'color_{node}') for node in nodes}
# 添加约束：相邻的节点不能有相同的颜色
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]  # 示例边
for u, v in edges:
    model.Add(color_vars[u] != color_vars[v])
# 定义目标函数（这里我们没有具体的目标函数，只是要找到可行解）
model.Minimize(0)
# 求解模型
solver = cp_model.CpSolver()
status = solver.Solve(model)
# 输出结果
if status == cp_model.OPTIMAL:
    print("最优解:")
    for node in nodes:
        print(f"节点 {node} 的颜色: {solver.Value(color_vars[node])}")
else:
    print("没有找到最优解")
```
##### 3. `PyGraphviz`
`PyGraphviz` 是 `Graphviz` 的 Python 接口，主要用于图形可视化和操作。虽然它本身不直接解决图着色问题，但可以与其他库结合使用来完成这项任务。
```python
import pygraphviz as pgv
import matplotlib.pyplot as plt
# 创建图
G = pgv.AGraph(directed=False)
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
# 绘制图
G.layout(prog='dot')
G.draw('graph.png')
# 显示图
img = plt.imread('graph.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```
### 网络流
*有些巨难的题我们就别想了*
*简单的问题倒是很简单,不需要笔记*
### 旅行商（TSP）问题
*目前还没有求解旅行商问题的有效算法*
详见[[example 2]]
### 计划评审法和关键路线法
*又叫统筹方法, 或PERT/CPM*
#### 计划网络图的概念
任何消耗时间或资源的行动称为作业。称作业的开始或结束为事件， 事件本身不消耗资源。 
在计划网络图中通常用圆圈表示事件，用箭线表示工作，如图所示，1、2、3表示事件，A、B表示作业。由这种方法画出的网络图称为计划网络图。
![[Pasted image 20240825152441.png]]
虚作业用虚箭线"$·····\rightarrow$"表示。它表示工时为零，不消耗任何资源的虚构作业。其作用只是正确表示工作的前驱后继关系。
##### 建立计划网络图应注意的问题
- (1) 任何作业在网络中用唯一的箭线表示，任何作业其终点事件的编号必须大于其起点事件。 
- (2) 两个事件之间只能画一条箭线，表示一项作业。对于具有相同开始和结束事件的两项以上的作业，要引进虚事件和虚作业
- (3) 任何计划网络图应有唯一的最初事件和唯一的最终事件。
- (4) 计划网络图不允许出现回路
- (5) 计划网络图的画法一般是从左到右，从上到下，尽量做到清晰美观，避免箭头交叉
##### 例图
![[Pasted image 20240825155455.png]]
![[Pasted image 20240825155520.png]]
之后使用图论方法(最长路)求出作业的关键路径
*关键路径就是1到8的最长路(1→3→5→6→8)*

