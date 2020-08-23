---
layout: post
title: 一站式深度学习基础
category: DeepLearning
tags: deep learning
description: basic knowledges among deep learning field
---

<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
</head>


## 1 Base Knowledge

此部分涉及到的知识有一定的理解门槛，如果基础知识有遗忘或者掌握不牢，参见一站式机器学习前部背景知识部分，此文不再重复。

### 1.1 Loss Func

**01 Loss**
$$
L(Y, f(x)) =
\begin{cases}
	1, Y\neq f(x) \\
    0, Y=f(x)
\end{cases}
$$

**MSE**

$$
L(Y,f(x)) = {(Y-f(x))}^2
$$

平方损失函数即均方误差(MSE)又叫做L2损失，对异常值敏感，某一个异常值可能导致损失函数波动太大。

**MAE**

$$
L(Y, f(x)) = |Y-f(x)|
$$

绝对损失函数即平均绝对误差(MAE)又叫做L1损失，导数可能不连续,基于上面两个损失函数各自的优缺点，有人提出改进的huber loss function

**Huber Loss Function**

$$
L_\delta(y,f(x)) = 
\begin{cases}
	\frac{1}{2}{(y-f(x))}^2, for|y-f(x)|\leq\delta \\
	\delta|y-f(x)|-\frac{1}{2}\delta^2, otherwise
\end{cases}
$$

只有在误差小于某个数时，才使用MSE，避免损失函数被异常点支配。

**条件熵**

两个随机变量x,y的联合熵

$$
H(x,y) = -P(x,y)logP(x,y)
$$

在x发生的条件下，y带来的新的熵称为x的条件熵H(Y\|X) 衡量已知随机变量x的情况，y的不确定性H(Y\|X) = H(X,Y) - H(X)​

![img](https://uploadfiles.nowcoder.com/images/20190315/311436_1552628862555_DBA6F761056A8FA361E96F7E44D51F7B)

**互信息**

$$
I(x,y) = \sum p(x,y)log^{\frac {p(x,y)}{p(x)p(y)}}\\
I(x,y) = D(p(x,y)||p(x)p(y))
$$

**交叉熵**

$$
L = -[ylog\hat{y} + (1-y)log(1-\hat{y})]
$$

二分类下当y = 1

$$
L = -log\hat{y}
$$

当y = 0

$$
L = -log(1-\hat{y})
$$

两者的函数图像都是接近正确分类时，损失函数越小，而且隔得越远，交叉熵越大

注意：**一般用神经网络解决多分类问题，会在输出层设置和类别数量一致的节点数，交叉熵衡量的是两个概率分布之间的相似度，label是概率分布，但是神经网络的输出不是概率分布，需要用softmax把神经网络的输出映射到概率分布**

交叉熵可以从两个方向去理解：

1 我们如果设sigmoid预测输出为y且真实类别也是y的概率定义表示为

$$
P(y|x) = \hat{y}^y*({1-\hat{y}})^{1-y}
$$

取对数之后和交叉熵就差一个负号，我们想最大化这个概率，等价于最小化这个概率的负数于是就得到交叉熵损失函数

2 从KL散度推导到交叉熵损失函数

相对熵又称KL散度,如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异，然后就自然的发现交叉熵其实就是KL散度的第二部分，以为KL散度表征是两个概率分布的差异，所以越小越好，自然的KL散度第一部分固定，那么最小化交叉熵就好

![kl1](/assets/img/deeplearning/one-stop/KL.png)

![WeChat Screenshot_20190812224516](/assets/img/deeplearning/one-stop/KL2.png)

**Reference** <br>
[简单的交叉熵损失函数，你真的懂了吗？](https://zhuanlan.zhihu.com/p/38241764)<br>
[一文搞懂交叉熵在机器学习中的使用，透彻理解交叉熵背后的直觉](https://blog.csdn.net/tsyccnh/article/details/79163834)

### 1.2 Activation Func

激活函数的发展过程：Sigmoid -> Tanh -> ReLU -> Leaky ReLU -> MaxOut

**sigmoid** 

函数图和导数图

![sigmoid](/assets/img/deeplearning/one-stop/sigma-sigma-prime.jpg)

**tanh**

函数图和导数图

![tanh](/assets/img/deeplearning/one-stop/tanh-tanh-prime.jpg)

**sigmoid和tanh的区别/联系**

Sigmoid 和 tanh 两个函数非常相似，具有不少相同的性质。简单罗列如下

- 优点：平滑
- 优点：易于求导
- 缺点：幂运算相对耗时
- 缺点：导数值小于 1，反向传播易导致梯度消失（Gradient Vanishing）

对于 Sigmoid 函数来说，它的值域是 (0,1)，因此又有如下特点

- 优点：可以作为概率，辅助模型解释
- 缺点：输出值不以零为中心，可能导致模型收敛速度慢

**这里有一个很重要的问题，为什么不为0中心，就导致收敛慢？**

在梯度更新的时候需要对sigmoid（w\*x+b）对w求导， => (w\*x+b)' \* sigmoid(f(x))' => sig(f(x))\*(1-sig(f(x)))\* x， 除x项外，其他都是常数，且恒大于0，从第二层起，神经元进过激活函数的输出都是正值，所以后续在做BP的时候，梯度恒为正。

**ReLU**

![relu](/assets/img/deeplearning/one-stop/relu.png)

对比sigmoid类函数主要变化是：

- 单侧抑制
- 相对宽阔的兴奋边界
- 稀疏激活性

存在问题：

ReLU单元比较脆弱并且可能“死掉”，而且是不可逆的，因此导致了数据多样化的丢失。通过合理设置学习率，会降低神经元“死掉”的概率。

**Leaky ReLU**

![leaky-relu](/assets/img/deeplearning/one-stop/leaky-relu.png)

优缺点：

- 避免ReLU神经元死亡的问题
- 能有负值输出
- 输入小于0时权重值是超参数

**MaxOut**

$$
Maxout(x) = max(w_1^T + b_1, w_2^T + b_2)
$$

**特殊的SoftMax**

_softmax一般只作为多分类的模型的最后一层的激活函数_，一般多分类模型的最后一个层的节点数就是类别数，把最后一层的输出总和归一，其中最大的节点值对应的类别就是预测的类别

$$
S_i = \frac{e^i}{\sum_{j}e^j}
$$

原理很简单，重点在于softmax如何进行梯度更新, 损失函数一般采用交叉熵

$$
Loss = -\sum_iy_ilna_i
$$

在多分类问题中，N长的label向量其中只有1个1其他全都为0，所以交叉熵中的求和可以忽略掉

![softmax-hand1](/assets/img/deeplearning/one-stop/softmax-hand1.jpg)

![WeChat Image_20190819163357](/assets/img/deeplearning/one-stop/softmax-hand2.jpg)

上面两个图片中有一个问题，为什么节点4要对w5i w6i 求偏导，他们又没有直接连接，因为softmax把节点456连接了起来，所以需要考虑节点4对56的影响；不然网络一次只更新一部分权重

从上面的推导可以看出，对于label为1节点相连的权重求偏导的时候g = (a-1)\*o; 对于其他权重g = a\*o

**Reference**<br>
[常见激活函数的比较](https://zhuanlan.zhihu.com/p/32610035)<br>
[详解softmax函数以及相关求导过程](https://zhuanlan.zhihu.com/p/25723112)

### 1.3 Optimizer

有大佬总结了不同的优化算法，其实他们都是符合一个框架的，设损失函数是$J(\theta)$,学习率是$\eta$

1. 计算损失函数关于当前参数的梯度 $g_t = \triangledown J(\theta_t)$
2. 计算一阶二阶动量$m_t$ ,$V_t$，即为一阶/二阶指数加权移动平均值
3. 计算当前下降梯度 $\triangle\theta_t = -\eta\cdot\frac{m_t}{\sqrt{V_t}}$
4. 参数更新 $\theta_{t+1} = \theta_t+\triangle\theta_t$

**SGD**(Stochastic Gradient Descent)没有动量的概念，所以一阶动量$m_t = g_t$, 二阶动量为“1”。
$$
\triangle\theta_t = -\eta \cdot g_t \\
\theta_{t+1} = \theta_t -\eta \cdot g_t
$$
SGD的缺点很明显，容易陷入局部最优解，因为只考虑当前的梯度，如果遇到局部最优点梯度为0，参数不继续更新，陷入局部最优而找不到全局最优。而且只考虑当前梯度的话，优化的路径方向波动太大，收敛速度慢。

**Momentum**(SGD with Momentum)引入“惯性”的概念，在SGD的基础上加上一阶动量，即考虑之前的优化方向，对当前梯度方向进行一个“修正/约束”。此时二阶动量为“1”，一阶动量是该时刻梯度的指数加权移动平均值.
$$
\triangle\theta_t = -\eta\cdot m_t = -(\eta g_t + \beta m_{t-1}) \\
\theta_{t+1} = \theta_t -(\eta g_t + \beta m_{t-1})
$$
**NAG**(Nesterov Accelerated Gradient) momentum算子中当前梯度方法主要由“惯性”决定，有助于帮助跳出局部最优点，NAG在该思想的基础上更加的“极端”,在计算当前梯度的时候，干脆就依照“惯性”方向再“走一步”计算下一步的梯度方向，没有考虑二阶动量为“1”
$$
\triangle\theta_t = -\eta\cdot m_t  =  -(\eta \triangledown J(\theta_t-\beta m_{t-1}) + \beta m_{t-1}) \\
\theta_{t+1} = \theta_t -(\eta \triangledown J(\theta_t-\beta m_{t-1}) + \beta m_{t-1})
$$
**AdaGrad** 引入二阶动量，之前的算子学习率都是固定的，二阶动量的引入可以动态改变学习率.没有考虑一阶动量，训练时历史累计梯度平方和$v_{t+1}$会越来越大，分母大，分子不变，学习率越来越小，单调递减。
$$
V_t = diag(v_{t,1},v_{t,2},...,v_{t,d}) \in R^{d\times d} \\
\triangle\theta_t = -\frac{\eta}{\sqrt{V_t + \epsilon}}\cdot g_t \\
\theta_{t+1} = \theta_t-\frac{\eta}{\sqrt{V_t+\epsilon}}\cdot g_t
$$
**RMSProp/AdaDelta** adagrad的缺点很明显，学习率单调递减，不管迭代情况如何，更新速度只会越来越慢。改进不累计全部历史梯度，只关心一个时间窗口,借用Momentum指数加权移动平均的思想
$$
v_{t,i} = \beta v_{t-1,i} + (1- \beta)g_{t,i}^2  \\
V_t = diag(v_{t,1},v_{t,2},...,v_{t,d}) \in R^{d\times d} \\
\triangle\theta_t = -\frac{\eta}{\sqrt{V_t + \epsilon}}\cdot g_t \\
\theta_{t+1} = \theta_t-\frac{\eta}{\sqrt{V_t+\epsilon}}\cdot g_t
$$
**Adam** 同时考虑一阶动量和二阶动量,观察一下公式就可以知道Adam就是RMSProp加了Momentum
$$
m_t = \beta_1m_{t-1} +(1-\beta_1)g_t \\
v_{t,i} = \beta_2 v_{t-1,i} + (1- \beta_2)g_{t,i}^2  \\
V_t = diag(v_{t,1},v_{t,2},...,v_{t,d}) \in R^{d\times d} \\
\hat{m_t} = \frac{m_t}{1-\beta_1^t} \\
\hat{v_{t,i}} = \frac{v_{t,i}}{1-\beta_2^t} \\
\theta_{t+1} = \theta_t-\eta\frac{\hat{m_t}}{\sqrt{\hat{v}+\epsilon}}
$$
**Nadam** 本质是adam考虑了NAG的思想，考虑未来梯度, Adam公式如下
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{V_t}+\epsilon}}\cdot \hat{m}_t \\
 = \theta_t- \frac{\eta}{\sqrt{\hat{V_t}+\epsilon}}\cdot(\frac{\beta_1m_{t-1}}{1-\beta_1^t} + \frac{(1-\beta_1)g_t}{1-\beta_1^t})
$$
计算t时刻的梯度，使用了t-1时刻的动量，如果我们用t时刻的动量近似代替t-1时刻，就引入了“未来因素”
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{V_t}+\epsilon}}\cdot \hat{m}_t \\
 = \theta_t- \frac{\eta}{\sqrt{\hat{V_t}+\epsilon}}\cdot(\frac{\beta_1m_{t}}{1-\beta_1^t} + \frac{(1-\beta_1)g_t}{1-\beta_1^t})
$$
**Reference**<br>[深度学习中的优化算法串讲](https://zhuanlan.zhihu.com/p/112381956)<br>[Adam 优化算法详解](https://baijiahao.baidu.com/s?id=1668617930732883837&wfr=spider&for=pc)

### 1.4 梯度爆炸/消失

梯度爆炸和梯度消失在某种程度上其实是一回事，在BP过程中，如果激活函数的梯度大于1，那么经过链式求导法则之后，梯度值会以指数形式增加，发生梯度爆炸，反之会以指数形式缩小，发生梯度消失。

解决梯度消失的两个办法

1: 使用 ReLU、LReLU、ELU、maxout 等激活函数

sigmoid函数的梯度随着x的增大或减小和消失，而ReLU不会。

2: 使用批规范化

通过规范化操作将输出信号x规范化到均值为0，方差为1保证网络的稳定性。从上述分析分可以看到，反向传播式子中有w的存在，所以w的大小影响了梯度的消失和爆炸，Batch Normalization 就是通过对每一层的输出规范为均值和方差一致的方法，消除了w带来的放大缩小的影响，进而解决梯度消失和爆炸的问题。

### 1.5 为什么使用交叉熵作为损失函数？

**特别注意，这里是说最后输出层的损失函数和激活函数，输出层的激活函数，和hidden layer的激活函数是分开设置的**要说为什么采用交叉熵，就要先看看为什么不用均方误差

均方误差

$$
loss = 0.5*(y-a)^2
$$

a是激活函数的输出结果，如果是sigmoid 

$$
a = sigmoid(z)
$$

其中

$$
z = w*x +b
$$

loss 对w求偏导,链式求导法则可知。 

$$
loss^, = (a-y)a * (1-a)*x
$$

**这其中(a-y)是损失函数导数，a\*(1-a)是激活函数的导数，x是线性函数的导数**，最大值才0.25,当label = 0 predict 接近1 或者label = 1,predict 接近0 的时候，sigmoid的导数值都很小，导致更新很慢，假设一种情况，初始化的时候很极端，正样本的预测值都很接近0，负样本的预测值都很接近1，这种情况下应该是非常错误的，应该马上大跨步的更新权重，但是如果是均方差loss func 并且采用sigmoid，梯度更新很慢。如果采用交叉熵作为损失函数，上面的loss对w求偏导只需要改一下loss函数的导数即可, 交叉熵的导数是
$$
crossentropy^, = (a-y)/a(1-a)
$$

和 sigmoid的导数相乘正好消除掉分母的影响，loss 对w的导数变成，这个时候predict 和label的差越大，梯度更新越快（起码最后一层的梯度更新快）

$$
(a-y)*x
$$

**Reference**<br>
[使用神经网络解决分类问题时，为什么交叉熵误差更好](http://heloowird.com/2017/03/08/diff_errors_of_neural_network/)

### 1.6 输出层损失函数和激活函数选择

二分类： sigmoid+交叉熵<br>
多分类：softmax+交叉熵<br>
回归：线性激活函数+均方误差<br>

**输出层的激活函数是按照任务来定的，二分类就是sigmoid，多分类是softmax，回归是线性激活函数**，但是在hidden layer中，为了抑制梯度消失，一般采用Relu。当sigmoid/softmax作为最后一层激活函数的时候为了让最后一层也可以加速梯度更新，抑制梯度消失，一般使用交叉熵作为损失函数。

### 1.6 Batch Normalization

**为什么想要去做BN？**

神经网络，特别是深度神经网络，随着梯度更新，各层权重w的值会增大，深层的网络激活函数输出已经到了饱和区（使用饱和激活函数例如sigmoid），浅层的梯度更新缓慢；称为Internal Covariate Shift(ICS)问题

解决梯度饱和问题的思路有两种

- 更换激活函数-Relu，Leaky Relu
- 从激活函数的输入分布入手，修正输入分布，使之落在饱和激活函数的敏感区（0值区附近）

BN层就是从第二个角度出发的。

**算法思路**

首先要说的是Batch Normalization是基于Mini Batch的，并且作用于激活函数之前, 对于每一个mini-batch的中的数据的每一个特征，基于特征粒度的z-score normalization，除数+一个基数是为了避免方差为0；

$$
\hat{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

这样处理之后，激活函数的输入是均值为0，方差为1的N个特征值；缓解了梯度饱和的问题，使得输入落在激活函数的敏感区间，**可是这样带来了一个更大的问题，经过normalization后的数据的表达能力下降**，为什么？因为均值为0，方差为1 的输入会落在sigmoid函数的0值附近，**进入了非线性激活函数的线性区域，丧失了非线性的表达能力**

因此，BN又引入了两个**可学习（learnable）的参数 $\gamma$与$\beta$** 。这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换，$\widetilde{Z_j}=\gamma \hat{Z_j}+\beta_j$特别地，当$\gamma^2=\sigma^2, \beta=\mu$时可以实现等价变换（identity transform）并且保留了原始输入特征的分布信息。除非完全等价还原，否则一定损失了部分表达能力。用数据的表达能力换取模型的更快拟合。

整个过程就是通过normalization把各式各样的分布拉回到标准正态分布，设置一个全局的base，底色，均值为0，方差为1的标准正态分布，然后通过反拉伸和反偏移，把数据分布往原来的方向推回一部分，在表达能力和拟合速度之间取得一个平衡

**预测时候BN层的使用**

预测的时候没有办法计算预测样本的均值和方差，就**采用所有训练样本的均值和方差**，这是对样本分布的真实均值和方差的无偏估计

**BN层总结：**

- **使得每一层的输入数据的分布相对稳定，加快模型的拟合速度**，深层不需要适应浅层的权重变化，层与层之间解耦合。每一层单独学习
- **使得模型对参数不敏感**，由于BN层的加入，输入值相对稳定，上一层权重的比例缩放变化在下一层没有体现。避免浅层网络参数的变化蝴蝶效应般的影响到后面的深层网络
- **BN层允许模型使用饱和激活函数，缓解梯度消失问题**
- **BN层有一定的正则化效果，抑制过拟合**；对于任意一个样本，和不同的其他样本组合成mini-batch，它自己变化后的值都是不同的，相当于给模型添加了噪声，抑制过拟合

**Reference**<br>
[详解softmax函数以及相关求导过程](https://zhuanlan.zhihu.com/p/25723112)<br>
[Batch Normalization原理与实战](https://zhuanlan.zhihu.com/p/34879333)

## 2 DNN

### 2.1 Perceptron

感知机模型对应于特征空间中将实例划分为正负两类的分离超平面，故而是判别式模型

![perceptron](/assets/img/ML/one-stop-machine-learning/perceptron.png)

原始perceptron采用的激活函数是单位阶跃函数，value set {+1，-1}, 由于感知机模型的输出是0和1两个离散的值，如果使用基于分类错误的平方误差，会使得损失函数不连续，更别说是否可导了。所以这里使用下面这个损失函数,该函数在SVM模型中被称为函数间隔 margin

$$
E= -\sum_{n\in M}w^T\phi(x_n)t_n
$$

其中，M 表示被分类错的样本集，t 表示样本的原始类别，∅(x) 表示经过处理后的输入，w\*∅(x) 表示在经过activation function之前的矩阵点乘结果  由于M 是分错类的样本集，w\*∅(x) 和 t 始终异号，结果始终大于零，所以损失函数就是\|w\*∅(x)\| 求和，是一个连续值， 且是凸函数，凸函数可以利用梯度下降法求解，需要求解什么，就对什么求梯度。

![perceptron-gradient](/assets/img/ML/one-stop-machine-learning/perceptron-gradient.png)

![perceptron-gradient1](/assets/img/ML/one-stop-machine-learning/perceptron-gradient1.png)

由上式可以看出，下一次迭代时的权重，由上一次的权重加上学习率加权过的全部输入结果的总和（input set 是分类错的样本集），是明显的batch training，由于巨大的计算量，可以改进为随机梯度下降方法，随机取M中的一个进行梯度下降，此时的梯度下降方法跳跃很大，但是总体上是往最优值跳跃的

![perceptron-sgd](/assets/img/ML/one-stop-machine-learning/perceptron-sgd.png)

每当有分类错误点，权重更新使得分类面朝分类错误点移动,感知机收敛的条件是训练集是线性可分的，如果线性不可分，那么感知机训练过程将永远不会收敛。感知机一旦训练到没有分类错误点就停止了，也就是即是刚刚移动到一个满足全部分类正确的位置，就停止了，没有进行最优化判断，不同的初值会影响最后的分类面。

**Reference**<br>
[感知机](https://www.zybuluo.com/Duanxx/note/425280)

### 2.2 AutoEncoder

autoencoder是前馈神经网络的一种,通常用来降维和特征提取. autoencoder并不关心模型最后输出和真值之间的差距大小，也就是损失函数的大小,重点关注的是hidden layer,这是一个典型的“**Fake Task**”, autoencoder原始结构就是一个只含有一个hidden layer的神经网络，当然隐层的数量可以增加。通常hidden layer的维度要远小于input layer，output layer 是用来还原input layer的，模型训练目标是尽可能的使input layer和output layer差异小。此时我们对输入的一个sample取对应的hidden layer的输出就是该sample降维之后的表达, hidden layer的权重矩阵就是embedding矩阵。

![auto-encoder](/assets/img/deeplearning/one-stop/auto-encoder.png)

**Reference**<br>
[当我们在谈论 Deep Learning：AutoEncoder 及其相关模型](https://zhuanlan.zhihu.com/p/27865705)

## 3 Time Series Model

### 3.1 RNN

RNN能给传统的神经网络带来了时间维度的考虑，传统的DNN 假设输入之间是完全独立的。假定现在有一个分类的需求，一个球需要判断是往右滚还是往左滚，输入的label是球的坐标，在不给定时间序列的情况下，球左右滚都有可能，但是一定给定时间上的先后顺序，那么球的滚动方向就确定下来了。另外人类的语音输入也是具有强烈的前后关联的，单独把每个词分开看，并不能或者很难看出语句的含义。

RNN则是在前馈神经网络上添加一个传递先前信息的循环，把上一个输入的hidden state 传递给下一个状态，但是如果状态数过多，太早的状态的信息由于梯度消失问题而消失，所以RNN只具有短期记忆

### 3.2 LSTM

谈到LSTM之前首先要说**Recurrent Neural Network**

\#to be completed

## 4 NLP

### 4.1 TF-IDF

TF(term frequency) = 词在文章中出现的次数/文章的总词数<br>
IDF(Inverse Document Frequency) = log(语料库总文章数/包含该词的文章数+1)<br>
TF-IDF = TF\*IDF

于是一个词的TF-IDF和词出现次数正比，和词出现的文章数反比，因为如果一个词在较少的文章中出现，那么这个词就更有区分度；注意**TF计算是针对这一篇文章，而IDF是针对整个语料库**

### 4.2 HMM

隐性马尔科夫模型(Hidden Markov Model)用来描述一个含有隐性变量的马尔科夫过程。与传统的马尔科夫过程不同的是，HMM除了具有显性的可见序列，还有隐性序列。在参考博客中有很详细的例子来描述HMM，在此借用一下大佬的描述。

假设现在我们手上有**3个不同的骰子**，4面，6面和8面分别用4,6,8表示。骰子质地均匀，**投出每一面的概率相同**，放入一个黑盒中。一次实验过程是，**随机抽出一个骰子**，投掷一次，然后放回盒子。进行多次重复实验。假设现在骰子点数组成的序列为1->2->6->8->5->4，骰子本身组成的序列可能是4->6->6->8->8->4，现在把这个过程抽象一下用来解释HMM的五元组。

| 五元组          | comments                                                     |
| --------------- | ------------------------------------------------------------ |
| 观测序列obs     | 骰子点数组成的序列                                           |
| 隐性状态states  | 骰子本身                                                     |
| 初始概率init_p  | 初始状态的概率，例子中是相同概率1/3                          |
| 转移概率trans_p | 从当前状态转移到下一个状态的概率，例子中是相同的1/3，通常用N\*N，N为骰子数的矩阵来描述 |
| 发射概率emit_p  | 给定骰子下，掷出每一面的概率。例子中是骰子均匀，掷出每一面概率相同，通常用N\*N，N为骰子能取的不同的点数个数的矩阵来描述 |

HMM根据缺失和已知信息的不同，大体上求解的问题可以分成三类。

- 知道隐性状态数量(骰子有几种)，知道转移概率，已知观测序列的情况下，求得隐性状态链。
- 知道隐性状态数量，知道转移概率，知道观测序列，想知道观测序列的发生概率

**第一类问题解法**

可以用最大似然状态路径去求解，最大化P(B\|A), A是观测序列，B是参数组合，也就是隐性状态链，我们想知道在显性状态链已经发生的情况下，什么样的隐性状态链产生这个显性状态链的概率最大。为了完成这个任务，也有两种解法，一种是暴力求解。遍历每一种隐性状态链，然后计算在该状态链下产生观测序列的概率。这种解法的计算量随着观测序列的长度指数上升。另一种方法是很有名的**Viterbi algorithm**，这是一种求解最短路径(最长路径可以转换为最短路径)的方法。以观测序列1->2->6->8->5->4来一步一步解释。

step 1 观测序列只有‘1’，很显然4面骰子投出1的概率最大，于是当前隐性状态链是‘4’

step 2 观测序列是‘1->2’，此时就需要计算第一个骰子分别是4,6,8的情况下，第二个骰子也分别是4,6,8的情况下那种组合的产生观测序列的概率最大，由于此时所有转移概率相同为1/3，整个问题简化为了贪心问题，求第二个骰子的最大概率就行。显然还是4面骰子产生2的概率最大，当前隐性状态链是‘4->4’。但是**当转移概率不同的时候，是需要考虑前面骰子的情况的**，但求解过程也不会很麻烦，因为上一步已经计算了到上一步为止各隐性状态的概率。

step 3 重复上述步骤，直到观测序列的最后一个节点，然后倒推出整个似然概率最大的隐性状态链。

**第二类问题解法**

依然可以暴力求解，遍历所有可能的隐性状态链，计算该状态链下产生观测序列的概率，所有概率相加就是结果。另一种方法是forward algorithm 前向算法。依然以观测序列1->2->6->8->5->4来一步一步解释。

step 1 当前观测序列是‘1’, 计算当前情况下产生‘1’的概率

| 骰子  | p1                     |
| ----- | ---------------------- |
| 4     | 1/3 \* 1/4             |
| 6     | 1/3 \* 1/6             |
| 8     | 1/3 \* 1/8             |
| total | sum(p1(4)+p1(6)+p1(8)) |

step 2 当前观测序列是‘1->2’，同样计算当前情况下产生2的概率

| 骰子  | p1       | p2                                 |
| ----- | -------- | ---------------------------------- |
| 4     | 1/3\*1/4 | 1/3\*1/6\* (p1(4) + p1(6) + p1(8)) |
| 6     | 1/3\*1/6 | 1/3\*1/6\*(p1(4)+p1(6)+p1(8))      |
| 8     | 1/3\*1/8 | 1/3\*1/8\*(p1(4)+p1(6)+p1(8))      |
| total | 0.18     | sum(p2(4)+p2(6)+p2(8))             |

step 3 当前观测序列是‘1->2->6’，继续。

| 骰子  | p1       | p2                                 | p3                            |
| ----- | -------- | ---------------------------------- | ----------------------------- |
| 4     | 1/3\*1/4 | 1/3\*1/6\* (p1(4) + p1(6) + p1(8)) | 1/3\*0\*(p2(4)+p2(6)+p2(8))   |
| 6     | 1/3\*1/6 | 1/3\*1/6\*(p1(4)+p1(6)+p1(8))      | 1/3\*1/6\*(p2(4)+p2(6)+p2(8)) |
| 8     | 1/3\*1/8 | 1/3\*1/8\*(p1(4)+p1(6)+p1(8))      | 1/3\*1/8\*(p2(4)+p2(6)+p2(8)) |
| total | 0.18     | sum(p2(4)+p2(6)+p2(8))             | sum(p2(4)+p2(6)+p(8))         |

step 4 以此类推计算到最后一个节点上的total值。

**Reference**<br>
[一文搞懂HMM（隐马尔可夫模型）](https://www.cnblogs.com/skyme/p/4651331.html)

### 4.3 CRF

每次理解新模型的时候，我都希望能通过一个简单易懂的例子来一步一步的从具体到抽象, 而不是上来就是硬推公式，疯狂劝退。好在参考博文的行文逻辑十分符合我的胃口，在此借用。

假设现在有一个分类问题，手上有一堆生活照，从早上到晚上都有，需要给照片打上标签，说明照片中的人是在做什么事情。我们当然可以直接根据单个照片的特征来进行分类，用一些已经标记好的照片训练一个模型，然后给未分类的照片打标签。这样固然可行，但是放弃了照片之间的时间关系，比如上一张照片在切洋葱，下一张照片人的眼睛闭上，那么大概率是因为洋葱辣眼睛才闭上的眼睛，而不是睡觉。

引申一下，CRF目前最合适的使用场景是词性标注，假设现在有一个句子是“I drank coffee at Timmy”，我们对每一个词打标签，I(名词) drank(动词) coffee(名词) at(介词) Timmy(名词)。当然有很多种标签组合，那么怎么知道哪一种组合更好呢？这就需要我们定义**特征函数**，或者说**特征函数集合**，给定标记序列，遍历所有特征函数，用每一个特征函数给标记序列打分，最后把分数加起来就是当前标记序列的总分数，分数最高的标记就是最好的标记序列。

一个仅考虑前后两个词性关联的线性特征函数f接受4个参数，输出是0/1，如果句子符合模式返回1，否则返回0, 先定义一些标志如下

- 句子$s$，待标记句子
- $i$，句子$s$中的第$i$个单词
- $l\_i$表示给第$i$个单词标记的词性
- $l\_i-1$表示给第$i-1$个单词标记的词性

给定特征函数$f\_i$一个权重$\lambda\_i$ ,那么句子$s$给定标记$l$的情况下分数等于

$$
score(l|s) = \sum_{j=1}^{m}\sum_{i=1}^{n}\lambda_jf_j(s,i,l_i,l_{i-1})
$$

外部求和是所有特征函数评分求和，内部求和是句子每个位置的单词特征值求和，然后经过SoftMax激活函数，就可以得到每一种标记的‘概率’

$$
p(l|s) = \frac{e^{score(l|s)}}{\sum_{\hat{l}}e^{score(\hat{l}|s)}}
$$

**Reference**<br>
[如何轻松愉快地理解条件随机场（CRF）？](https://blog.csdn.net/dcx_abc/article/details/78319246)

### 4.4 N-Gram

N-gram 是一种基于统计学的自然语言模型算法，对于一个字符串，它的n-gram就是按照长度N 滑动切割字符串得到的词段 。

**N-gram的主要应用**

**模糊匹配**<br>
两个字符串，计算他们的n-gram词段，从共有词段的角度去考虑他们的距离。以 N = 2 为例对字符串Gorbachev和Gorbechyov进行分段，可得如下结果

<u>Go</u>, <u>or</u>, <u>rb</u>, ba, ac, <u>ch</u>, he, ev <br>
<u>Go</u>, <u>or</u>, <u>rb</u>, be, ec, <u>ch</u>, hy, yo, ov

两个字符串的距离 8(第一个字符串的词段数)+9(第二个字符串的词段数)-2\*4(共有词段数) = 9

**评估句子是否合理**<br>
假设一个句子有m个词组成，则这个句子出现的概率为

$$
p(w_1,w_2,w_3,...w_m) = p(w_1)*p(w_2|w_1)*p(w_3|w_1,w_2)...p(w_m|w_1,w_2...w_{m-1})
$$

如果m很大，概率会非常的稀疏，且参数空间过于庞大，显然不好计算，于是利用马尔科夫假设，先验的认为一个词出现的概率只和前面的N个单词有关，即N-gram，和其他无关。

当N=1

$$
p(w_1,w_2,w_3...w_m) = \prod_{i=1}^{m}{p(w_i)}
$$

当N=2

$$
p(w_1,w_2, w_3...w_m) = \prod_{i=1}^{m}p(w_i|w_{i-1})
$$

当N=n

$$
p(w_1,w_2,w_3...w_n) = \prod_{i=1}^{m}{p(w_i|w_{i-1},w_{i-2}...w_{i-n})}
$$

计算一个句子出现的概率可以用句子的词段出现频率相乘去逼近, 计数值从熟料库中统计即可得。

$$
p(w_i|w_{i-1},w_{i-2}...w_{i-n}) =count(w_{i-1},w_{i-2}...w_{i-n},w_{i})/count(w_{i-1},w_{i-2}...w_{i-n})
$$

当N更大时约束力更强，辨识力更强，更加稀疏，因为N越大，语料库中出现该词段的概率越低； 并且n-gram词段总数随N指数增加，$V^n$ v是词汇总数。

**Reference**<br>
[自然语言处理中n-gram语言介绍](https://zhuanlan.zhihu.com/p/32829048)<br>
[自然语言处理中的N-Gram模型详解](https://blog.csdn.net/qq_21161087/article/details/78401469)

### 4.5 Word Embedding

自然语言中的单词计算机并不能理解它的意思，它只对应内存空间中的一串二进制码，所以在训练之前我们需要单词的‘另一种表达’，或者说一种编码，把单词映射到另一个容易被理解，容易后续处理的空间。那么人们自然就需要设计一个满足要求的映射关系来完成这个任务。

最简单的word embedding 方法是**基于词袋 (BagOfWords, BOW)**的one-hot表达方法，步骤如下：

1. 把语料中的单词去重整理并排序，排序方式自定义，称为词汇表
2. 对于一个句子中的单词，它的embedding结果就是一个和词汇表一样长的一个向量，向量中单词在词汇表中对应的index位置为1，其他为0
3. 句子中的每个单词按照步骤2的方法embedding之后，可以把所有单词的向量相加，构成句子或者文档的向量

基于词袋的方法主要有两个问题，第一个它仅仅考虑词是否出现，而不考虑前后顺序；第二个词袋可能十分的长，词向量极度稀疏。为了解决第一个问题，一个优化方法是**共现矩阵(cocurrence matrix)**, 其思想和n-gram类似，设定一个滑动窗口，在窗口内，任何两个词之间都属于共现过一次。例如I like machine learning and deep learning.假设滑动窗口为2，即考虑目标词前后两个词。

1. I \[like machine learning\] and math<br>
2. I like \[machine learning and\] math <br>
3. I like machine \[learning and math\] 

对于“learning”这个词来说，它与like, machine , and, math 共现过，且和mechine共现过两次，那么我们建立一个M\*M的矩阵，即共现矩阵，m是所有词汇的数量，矩阵中的值是两个词共现的数量，很显然这个矩阵是对称的，一列或者一行就是一个词对应的编码向量，这个向量的长度依然等于总的词汇数，而且也极度稀疏，那能不能用一个**连续短稠密向量**去刻画word呢？当然是可以的，这就是大名鼎鼎的word2vec模型，这个稠密向量被称为word的**Distributed Representation**

**Word2Vec**

从大量语料中以**无监督学习**的方法学习语义的模型，有别于传统的基于词袋的one-hot 编码()主要有两个模型，**CBoW(Continuous Bag-of-Words Model)** 和**Skip-gram**，两个模型的结构很相似，主要区别在于CBOW模型输入是context，输出是预测词；skip-gram输入是当前词，输出是预测的context。

**CBOW**模型结构如下	

![cbow](/assets/img/deeplearning/one-stop/cbow.jpg)

- 输入层，我们不能直接把自然语言的字符/字符串输入模型，需要字符数值型的“原始表达”，于是采用基于词袋的**one-hot encoder**，与N-gram中使用的方式相同
- 共享的embedding矩阵大小是V\*N，V是词典的总长度，N为自定义embedding向量的长度
- C个词向量乘上共享的embedding矩阵，**相加求平均**得到隐层的向量
- 隐层采用的**线性激活函数**，其实相当于没有激活函数
- 隐层输出乘上输出权重矩阵，得到1\*V 维度的输出向量，经过softmax层，概率最大的index所指的就是预测的目标词
- 输出和Truth求交叉熵作为损失函数

这个模型和其他模型的不同点在于，我们根本不关心模型预测的准确率，我们只需要模型训练过程中的副产品，副产品有两个，一个是输入层到隐层的embedding矩阵，一个是隐层输出到输出层的输出层矩阵，两个矩阵转置之后维度一样，一般我们取embedding 矩阵作为**词向量**，embedding矩阵的大小V\*N，而且输入的词编码只有一个index为1，其他为0，所以词向量乘上embedding矩阵相当于把取值为1的index对应的embedding vector取出来, 但是该神经网络的问题也很明显，参数量太大，embedding矩阵大小是V\*N, 通常来说词典的长度都是百万级以上，那么该矩阵参数轻轻松松上千万甚至亿级别，训练起来会是一场灾难。有负采样和层级softmax可以缓解这个问题，负采样是指梯度更新的时候只更新一部分参数，例如只更新当前context出现的词对应的词向量，但是输出层计算softmax时需要对所有词典都计算概率值，计算量非常大。层级softmax占坑\# to be completed，后面再更。

**为什么以条件概率为优化目标最后训练出的embedding 矩阵就能代表词向量呢？**

这个问题其实很重要，但是网上绝大多数博客都没有提到过，大多只是介绍了CBOW和skip-gram而从来没有考虑模型训练结果的有效性。以下是个人理解，比如现在句子是I like deep learning， 目标词是deep，context是I, like, learning，为什么I, like, learning的词向量取平均训练出的embedding matrix就能给出合理的deep的词向量，那是因为训练有很多个sample，在其他sample中，deep作为context输入去预测其他词，这个时候模型会拟合出它觉得“合适”的deep的词向量。也就是说deep的词向量主要是当deep为context的时候学习到的，而不是作为目标词的时候学习到的。

**Skip-Gram** 看上去是cbow模型逆转因果的结果，但是细节上又有些让人容易迷惑的地方，首先模型结构是

![skip-gram](/assets/img/deeplearning/one-stop/skip-gram.png)

在模型训练时，我们选择一个词作为输入词，同样的输入的词的one-hot形式的表达，定义skip_window参数，意思是我们取输入词前后各多少个词参与训练，图中我们选love作为中间词，skip_window选2.那我们分别要求出

$$
p(w_{Do}|w_{Love}), p(w_{you}|w_{Love}), p(w_{deep}|w_{Love}), p(w_{learning}|w_{Love})
$$

假设我们的语料一共长度为V，skip_window为2，则训练过程一共需要预测V\*4个条件概率(忽略开头和结尾少去的几个)。**需要注意的是这个时候‘Do’，‘you’， ‘deep’， ‘learning’相对于love的顺序已经不重要了，需要预测就是在love出现的情况下，出现其他四个单词的概率，而不是给定love的情况，其他四个单词出现在指定位置上的概率**

![](/assets/img/deeplearning/one-stop/skip-gram-sample.png)

所以这就是图上迷惑人的地方，**模型结构图感觉起来像是输入词的embedding向量乘上了很多次输出权重矩阵，实际上只需要乘一次就可以知道所有的条件概率**，给定中间词，生成背景词的条件概率就是

$$
P(w_{do}, w_{you}, w_{deep}, w_{learning}| w_{love}) = p(w_{Do}|w_{Love})*p(w_{you}|w_{Love})*p(w_{deep}|w_{Love})*p(w_{learning}|w_{Love})
$$

因为假定了背景词之间是**独立的**, 训练目标就是最大化该概率。

**Reference**<br>
[超详细总结之Word2Vec（一）原理推导](https://blog.csdn.net/yu5064/article/details/79601683)<br>
[秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)<br>
[NLP之——Word2Vec详解](https://www.cnblogs.com/guoyaohua/p/9240336.html)<br>
[一文详解 Word2vec 之 Skip-Gram 模型（结构篇）](https://blog.csdn.net/qq_24003917/article/details/80389976)<br>
[NLP之---word2vec算法skip-gram原理详解](https://blog.csdn.net/weixin_41843918/article/details/90312339)<br>
[Word2Vec介绍：直观理解skip-gram模型](https://zhuanlan.zhihu.com/p/29305464)<br>
[CBOW模型](https://www.jianshu.com/p/d2f0759d053c)<br>

## 5 Reinforcement Learning

### MCTS

**MCTS(Monte Carlo Tree Search)**即蒙特卡洛树搜索,是一种有效解决探索空间巨大问题的方案。为了说明的方便，限制“游戏”背景为**双人有限零和顺序游戏**，即玩家为两名，在任意时间点，玩家间的交互方式有限，玩家交替进行动作，最终游戏结果双方受益相加为零。如果游戏探索空间很小，例如井字游戏，3*3个可落子空间，棋盘状态满打满算3^9，完全在可接受的范围内，而且其中存在大量无效状态，有效状态数量只会更少。这个时候可以用游戏树来表示游戏。

**游戏树**的第一层为一个根节点表示游戏初始状态，第一层有9个节点表示9个落子空间都可以落子，每个第一层节点又有8个第二层子节点，表示在第一层落某个空间的情况下，剩下的8个可能落子点...以此类推直到完全探索完毕，树的总节点数就是游戏所有可能的状态数。如果搜索空间足够小，我们是可以穷尽游戏所有状态，也就是可以计算出每个节点获胜的概率，假设游戏树公开，双方的最优策略即从根节点开始，先手方选择获胜概率最大的落点，后手方选择先手方获胜概率最小的落子点，以此类推直至游戏结束，称为minmax策略。

当搜索空间足够大的时候，游戏树显然不合适，需要一种高效的探索方式，在无法穷尽所有节点的情况下，兼顾探索和利用，即MCTS, 适合使用MCTS方法的”游戏“需要满足**零和，游戏信息公开，交互结果确定，顺序执行，操作离散性**等调价，方法主要包含**Selection，Expansion， Simulation， Backpropagation** 四个阶段。

MCTS是如何兼顾探索和利用的呢？ MCTS的经典实现UCT(Upper Confidence Bounds for Trees)提到的UCB算法
$$
argmax_{v'\in children\_of\_v} \frac{Q(v')}{N(v')} + c\sqrt{\frac{2lnN(v)}{N(v')}}
$$
其中，$Q(v')$表示子节点当前累计quality值，$N(v')$表示子节点visit次数，式中第一项即表示子节点平均收益，$N(v)$ 表示父节点visit次数，式中第二项子节点visit次数越少，值越大，两项求和则同时兼顾了收益和探索。下面从MCTS的四个阶段详细说明

1. Selection 找到一个未被完全探索的节点，未被完全探索表示该节点至少有一个子节点没有经过探索，如果没有，就选择UCB值最大的节点，递归进行。
2. Expansion 选定节点之后，在该节点基础上，走出一步，创建一个新节点，通常是随机选择。
3. Simulation 在新创建的节点上进行模拟游戏，直到游戏结束，获得该expansion出的节点的reward。
4. Backpropagation 把reward更新到所有父节点中，包括quality value 和visit time，用于计算UCB。

参考Ref1知乎大神的代码，大神写的是python代码，本人改写为[scala版本](https://github.com/weiqi-jiang/DataStructure-Algorithm-Model/blob/master/scala/tree/MCTS.scala)，但效果一般，简单的游戏并不能保证每次都得到最优的结果，待优化。

**Reference**<br>[如何学习蒙特卡罗树搜索(MCTS)](https://zhuanlan.zhihu.com/p/30458774)<br>[蒙特卡洛树搜索（新手教程）](https://blog.csdn.net/qq_16137569/article/details/83543641)