---
layout: post
title: Data Structure& Algorithm
category: Resource
tags: algorithm
description: 常用的一些数据结构和算法，例如最短路径，BFS,DFS
---

## **Data Structure**

### Stack（栈）
- LIFO （LAST IN FIRST OUT)
- 就像是一根树在地上的竹筒，先进去的东西等到上面的东西拿走才可以拿走

### Queue(队列)
- FIFO （FIRST IN FIRST OUT）
- 就像水管，“水（data）”从固定一头进去，从另一头按照同样的顺序出来

### 循环队列
循环队列是队列存储结构中最常使用的，因为如果是普通队列，pop出头节点是需要把后续节点前移的，如果不前移，仅仅是使用front，rear指针会出现“假溢出”，pop 和插入的时候分别front+=1 rear+=1，会出现队列其实没有满，但是rear已经到队列地址的最后了，不能+1了。*front永远指向头部元素，tail用于指向队列最后一个元素的下一个地址*循环队列可以解决这个问题，当rear到尾部的时候，再插入的话，rear指向首地址
判断条件：

- 队列为空 front == rear
- 队列满： (rear+1)%maxsize == front
- 队列长度 (rear-front + maxsize)%maxsize
这里要特别说明为什么需要（rear+1)%maxsize == front
是因为要和队列为空区分开，不然不知道是队列满还是队列为空.所以一定是有一个节点是空出来的，**但是这个空余的节点不一定是首地址，不然会想不通上面的判断条件，一旦想通了空余节点是动态的，就没有问题了**

插入操作：
rear = (rear+1)%maxsize

弹出操作：
front = (front+1)%maxsize

**Reference**<br>[深入理解循环队列----循环数组实现ArrayDeque](https://www.jianshu.com/p/6b88855017d5)

### Heap(堆)
- 完全二叉树
- 某个节点的节点值总是大于等于或者小于等于子树的所有节点的值
- 堆中的所有节点的子树都是堆树
- 分为最大堆和最小堆 max heap， min heap
- 具体实现使用List列表，假设实现最小堆，实际操作中在heapList的*index为0的位置提前插入一个冗余节点*，方便后续index相关操作

  **插入**：首先在heapList中append（data），heapList虽然是list列表形式，但是可以看成一个二叉树结构，插入到二叉树最后一层的叶子节点上，之后进行上浮操作
  **创建堆**：对堆列表(heapList)前半节点进行sink操作(完全二叉树的后半节点都是叶子节点，没必要进行sink操作)
  **出堆：** 把根节点用最后一个叶子节点替代，根节点弹出，对新的根节点sink操作
  **上浮**：将该节点和父节点比较，如果父节点比该节点小，交换两个节点，循环该操作直到寻迹到根节点
  **下沉**：将该节点和两个子节点进行比较，如果父节点比子节点小，交互两个节点，循环直到到达叶子节点。存在两个叶子的情况下，交换叶子节点中较小的那一个，确保最小堆中某一个节点是以该节点为根节点的子树中最小的(以上详情参见github)

### Tree（树）

```python
"""
前中后序遍历的递归版本
- 中序遍历：左根右
- 前序遍历：根左右
- 后序遍历：左右根
- 层序遍历： BFS
"""
def InOrderTraversal(root,res):
    if not root:
        return 
    InOrderTraversal(root.left)
    res.append(root.val)
    InOrderTraversal(root.right)
    
def PreOrderTraversal(root,res):
    if not root:
        return 
    res.append(root.val)
    PreOrderTraversal(root.left)
    PreOrderTraversal(root.right)
    
def PostOrderTraversal(root,res):
    if not root:
        return 
    PostOrderTraversal(root.left)
    PostOrderTraversal(root.right)
    res.append(root.val)
"""
非递归版本
利用stack的特性来实现，不能使用queue！，不能用queue！，不能用queue！
"""
# 压栈先右后左，出栈先左后右，当前元素总是遍历
def PreOrderTraversal(root,res):
    stack = [root]
    if not root:
        return []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res
# 先while循环压cur.left进栈,栈顶出栈就包含了左中，如果node.right,cur置为node.right
# 然后while loop 压node.right.left 进栈
def InOrderTraversal(root,res):
    res = []
    stack = []
    cur = root
    while True:
        while cur:
            stack.append(cur)
            cur = cur.left
        if not stack:
            break
        node = stack.pop()
        res.append(node.val)
        if node.right:
            cur = node.right
    return res
# 双栈发
# 第一个栈入栈顺序左右根，出栈顺序根右左
# 第二个栈入栈顺序根右左，出栈顺序左右根
# 为什么需要两个栈，第一个栈入栈顺序就是左右根，为什么不用queue直接输出呢？
# 因为第一个栈的左右根顺序是当前节点的，不是整个树递归到叶子节点的“左右根”那么上层节点的“左右根”需要先压进栈，最后再输出
def PostOrderTraversal(root, res):
    stack1 = [node]
    stack2 = []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    while len(stack2) > 0:
		print(stack2.pop().val)
```

**Reference**<br>[Python 实现二叉树的前序、中序、后序、层次遍历（递归和非递归版本）](https://blog.csdn.net/m0_37324740/article/details/82763901)<br>

## Search Algorithm

### linear search
- best case: O(N)
- worest case: O(N)
- general case: O(N)

### Binary search
- best case: O(logN)
- worest case: O(N)
- general case: O(logN)

### BFS

```python
class TreeNode:
    def __init__(self, val=0,left=None, right=None):
        self.left = left 
        self.val = val
        self.right = right

"""
iterative form
BFS使用queue辅助实现
"""
def bfs_iterative(root):
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

"""
recursive form
一般BFS没有递归写法
"""
```

### DFS

```python
class TreeNode:
    def __init__(self, val=0,left=None, right=None):
        self.left = left 
        self.val = val
        self.right = right

"""
iterative form
DFS使用stack辅助实现， 先进后出
"""
def dfs_iterative(root):
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)
       	if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
"""
recursive form 
本质上也是stack来实现
"""
def dfs_recursive(root):
    print(root.val)
   	if root.left:
        dfs_recursive(root.left)
    if root.right:
        dfs_recursive(root.right)
```





## Graph 图论

### 邻接矩阵

用一个二维数组来存储表示图，一个一维数组保存顶点信息

设图G(V,E) Adjacent Matrix size: \|V\|\*\|V\|

| index |  0   |  1   |  2   |  3   |
| :---: | :--: | :--: | :--: | :--: |
|   0   |  0   |  1   |  1   |  0   |
|   1   |  1   |  0   |  0   |  1   |
|   2   |  1   |  0   |  0   |  0   |
|   3   |  0   |  1   |  0   |  0   |

邻接矩阵的性质：

1. 对称矩阵 am\[i\]\[j\] = am\[j\]\[i\]
2. 矩阵中*一行或者一列的总和对应vertex的顶点的度*
3. 如果G 有边权重，则 am\[i\]\[j\] = wij 当i，j之间有边存在，am\[i\]\[j\] = 0 当i==j; am\[i\]\[j\] = inf 当i.j 不存在边
4. 在生成的时候节点数就固定了，新增节点的开销很大，对于稠密图节约空间
5. 遍历的时间复杂度 O(\|V\|^2)

### 邻接链表
一个一维数组存储节点空间，每个顶点链接点构成一个单链表
```python
adjacent_linked_list = {
    node1: [node2,node3,node4],
    node2: [node1, node5,node6],
    # ...
}
```
邻接链表的性质：
- 对于边数相较于节点数很小的图来说，\|E\| << \|V\|\*\*2 具有很高的存储效率, 即邻接矩阵很稀疏
- 对于边带权重的情况，可以在存储结构上添加权重信息
- 时间复杂度 O(\|E\|+\|V\|) 所以如果\|E\| << \|V\|^2的话，不管空间复杂度节约，时间复杂度也节约

### 图的遍历和搜索
[BFS,DFS 代码实现](https://github.com/JIANGWQ2017/DataStructure/blob/master/graph.py)

**BFS**：常用来寻找最短路径

- queue辅助
- 一层一层的寻找，可以寻找到最短路径
- 由于G（V,E）并不是所有节点都连接成一起，为了保证遍历到图中的所有点，所以*需要对所有节点都进行BFS遍历*，维护一个visited list来表示节点是否访问过，不会重复访问
- 在任何一个BFS遍历中，首先把第一个节点放入进stack，**维护一个和节点数等长的list ，isinqueue，用来表示节点是否已经放进stack了（注意这里不是已经visited了，而是已经放进stack）**，isinqueue首节点的值设为1； cur = stack.pop(0)，visited[cur] = 1 遍历邻接点，把还没有放进stack的放进stack isinqueue对应位置设为1.
- 直到stack为空
- （其实这里的isinqueue，visited 就是常见博客中说到的节点的三种状态，黑白灰，白表示从来没有访问过，灰表示进栈但是还没有出栈，黑表示已经出栈）

BFS的时空复杂度：
- 如果采用链接链表，每一个节点都要访问，链接链表中每一个边都要访问（其实是访问两遍\|E\|）时间复杂度O(\|V\|+\|E\|)
- 如果是邻接矩阵，每一个节点访问，遍历邻接节点的时候又需要走\|v\|步，所以TC = O(\|V\|**2)
- 空间复杂度因为要维护一个isinqueue list, SC = O(v)

**DFS**: 不能用于寻找最短路径
- stack 辅助
- 一条路走到黑，再从最近的岔路开始又一条路走到黑，不能寻找到最短路劲
- 由于G(V,E)并不是所有节点都连接成一起，所以需要对所有节点都进行DFS遍历，维护一个visited list来表示节点是否访问过
- 在任何一个DFS遍历中，首先把第一个节点放入进stack，维护一个和节点数等长的list ，isinqueue，用来表示节点是否已经放进stack了（注意这里不是已经visited了，而是已经放进stack），isinqueue首节点的值设为1； cur = stack.pop()，visited[cur] = 1 遍历邻接点，把还没有放进stack的放进stack isinqueue对应位置设为1.
- 直到stack为空

DFS时间复杂度：

- 空间复杂度因为要维护一个isinqueue list, SC = O（v）
- 时间复杂度和BFS一样

**拓扑排序**： 

- 可以用来检测环的存在
- 必须是有向无环图DAG
- 只要是有向无环图一定存在拓扑排序，但是不一定只有一个

### 图中的环

**无向图**

**解法1：**如果是图是无向图就可以用BFS的方法判断有没有环，BFS中节点分为三个状态，白，灰，黑。对于一个节点，遍历它的链接点，如果有节点在isinqueue中对应位置为1 并且该节点不是父节点，那么就有环。于是需要在BFS的基础上，保存每一个节点的父节点信息

**解法2：**类似于有向图kahn算法

1. 计算每个节点的度，保存一个degree list
2. 对于每一个度<=1的节点，删除该节点，degree list中对应位置数值-1
3. 并删除相邻的边，也就是把邻接点的度-1, 实现中是通过邻接链表或者邻接矩阵找到相邻点，在degree list 对应位置值减一
4. 重复2-3步骤，直到没有可以更改的节点，如果现在还有节点degree 大于0，说明有环，否则没有环

**有向图**

**kahn算法，有向图拓扑排序**

- 计算图中所有点的入度，把入度为0的点加入栈
- 如果栈非空：
  - 取出栈顶顶点a，输出该顶点值，删除该顶点
  - 从图中删除所有以a为起始点的边，如果删除的边的另一个顶点入度为0，则把它入栈
- 如果图中还存在顶点，则表示图中存在环；否则输出的顶点就是一个拓扑排序序列

## 字符串匹配(BF-KMP)

### BF 算法（Brute Force）
其实就是暴力解，设str1 是被查找字符串，str2是pattern
```python
#初始
i，j = 0，0
while i < len(str1) and j<len(str2):
	if str1[i] == str2[j] : 
		i++,j++
	else: 
		i = i-j+1 #（指向原来开始匹配的下一个字符）
		j = 0
else:
	if j >=len(str2):
		return i-len(str2)
	else:
		return -1
```
时间复杂度网上说的是最好的情况 O(str + pattern) 最坏情况O(str\*pattern); 我个人觉得最好的情况是O(str)也就是一个字符都不匹配

**Reference**<br>[模式匹配——从BF算法到KMP算法](https://blog.csdn.net/ns_code/article/details/19286279)

## 背包问题

### **基本01背包问题**
背包重量最大H，n个物品，有对应的n个价格V，对应的n个重量w。 **状态：**value\[i\]\[h\] 表示把i个物品放进大小为h的背包能达到的最大价值; **初始化：**value 的size (n+1)\*(H+1)。**状态转移方程：**

```
value[i][h] = max(value[i-1][h], value[i-1][h-w]+V[i]) (if w<= h)
value[i][h] = value[i-1][h] (if w>h)
value[i][h] = 0 (if h or i == 0)
```
**思路:**对于每一个物品，有两个选择，加入还是不加入。如果加入，当前优化问题转换为子问题，因为物品i要加入，所以现在背包可用空间只有h-w，子问题就是在i-1个物品，在背包只有h-w下能达到的最大价值加上现在物品i的价值。如果不加入，那就是i-1个物品在背包大小为h下能取得的最大价值。比较加入还是不加入的价值，较大的那个就是当前状态能取得的最大价值

**优化点**：时间复杂度没有优化的点，但是空间复杂度有可以优化的。观察value\[i\]\[h\]当前的状态只和value\[i-1\]\[h\]的各个状态有关。如果我们只有一个数据value\[h\]来完成dp呢，我们必须保证在value\[h\]更新值的时候value\[h\],value\[h-w\]保存的是上一轮的值，不是更新后的值。对于给定的h, value\[h\]下一轮的值只可能与value\[g\]有关 g<h,于是我们只要从H ->0 的方向来更新value\[h\]就可以了

```python
for i in range(n):
	for h in range(H,-1,-1):
		value[h] = max(value[h], value[h-w]+V[i])
```
**如果要求必须装满呢？必须装满的前提下的最大值**，只需要在初始化的时候做一下改变。1维dp，就把value\[0\] = 0, value\[i\] = -INF i!=0; 2维dp，value\[i\]\[0\] = 0, 其他全部-INF

### **完全背包问题**
	todo

**Reference**<br>[背包问题九讲](https://www.kancloud.cn/kancloud/pack/70125)<br>[DP背包问题的 恰好装满 问题](https://blog.csdn.net/Ratina/article/details/87859525)

## 位图

首先需要明确的是位图算法**通常用来判断某个数据存不存在**。假设现在有个问题，有10亿个Int型，判断某个输入是否在这10亿个中存在。这个问题其实无关时间复杂度，主要问题在于内存是否能存下10亿个int型数据。针对大规模数据，位图能提供一种大幅节约空间复杂度的方法。通常位图用数组实现，数组的每一个元素的每一个二进制位都可以用来标识某一数据存在与否。假设现存数据最大为60，最小为0，可以用一个包含两个Int数值的数组来表示。设输入为x，x在范围[0, 60], 为了方便，设x=37，编码步骤如下：

1. int(37/32) = 1   商为1，表示落在第二个Int值内
2. 37%32 = 5  表示index为5的位置为1 

按照上述方法把存量都进行编码，对于待判断输入，也进行一次编码，查找对应bit位是否为1，如果是，则存在。

**实现方式**

**对应bit位置1**，源位图和1按位或的方式实现

![bitmap](/assets/img/resource/datastructure_algorithm/bitmap.png)

**对应bit位置0**，源位图和1的取反结果按位与的方式实现

![bitmap](/assets/img/resource/datastructure_algorithm/bitmap-reset.png)

**Reference**<br>[数据结构：位图](https://blog.csdn.net/etalien_/article/details/90752420)<br>[数据结构-位图原理及实现](https://blog.csdn.net/lucky52529/article/details/90172264)