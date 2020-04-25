---
layout: post
title: 一站式深度学习基础
category: DeepLearning
tags: deep learning
description: basic knowledges among deep learning field
---

## 基础知识

此部分涉及到的知识有一定的理解门槛，如果基础知识有遗忘或者掌握不牢，参见一站式机器学习前部背景知识部分，此文不再重复。

### 常见损失函数

![loss-func](/assets/img/deeplearning/one-stop/loss-func.png)

**MSE**

平方损失函数即均方误差(MSE)又叫做L2损失，对异常值敏感，某一个异常值可能导致损失函数波动太大。

**MAE**

绝对损失函数即平均绝对误差(MAE)又叫做L1损失，导数可能不连续。

基于上面两个损失函数各自的优缺点，有人提出改进的huber loss function

**Huber Loss Function**

![huber_loss](/assets/img/deeplearning/one-stop/huber_loss.png)

只有在误差小于某个数时，才使用MSE，避免损失函数被异常点支配。

**条件熵**

两个随机变量x,y的联合熵
$$
H(x,y) = -P(x,y)logP(x,y)
$$
在x发生的条件下，y带来的新的熵称为x的条件熵H(Y|X) 衡量已知随机变量x的情况，y的不确定性H(Y|X) = H(X, Y) - H(X)

![img](https://uploadfiles.nowcoder.com/images/20190315/311436_1552628862555_DBA6F761056A8FA361E96F7E44D51F7B)

**互信息**
$$
I(x,y) = \sum p(x,y)log^{\frac {p(x,y)}{p(x)p(y)}}
$$

I(X,Y)=D(P(X,Y)||P(X)P(Y))

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

1 我们如果设sigmoid预测输出为y且真实类别也是y的概率定义表示为![[公式]](https://www.zhihu.com/equation?tex=P%28y%7Cx%29%3D%5Chat+y%5Ey%5Ccdot+%281-%5Chat+y%29%5E%7B1-y%7D)，取对数之后和交叉熵就差一个负号，我们想最大化这个概率，等价于最小化这个概率的负数于是就得到交叉熵损失函数

2 从KL散度推导到交叉熵损失函数

相对熵又称KL散度,如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异，然后就自然的发现交叉熵其实就是KL散度的第二部分，以为KL散度表征是两个概率分布的差异，所以越小越好，自然的KL散度第一部分固定，那么最小化交叉熵就好

![kl1](/assets/img/deeplearning/one-stop/KL.png)

![WeChat Screenshot_20190812224516](/assets/img/deeplearning/one-stop/KL2.png)

**Reference：**

 https://zhuanlan.zhihu.com/p/38241764

https://blog.csdn.net/tsyccnh/article/details/79163834



### 常见激活函数的比较

激活函数的发展过程：Sigmoid -> Tanh -> ReLU -> Leaky ReLU -> MaxOut

**sigmoid** 

函数图和导数图

![sigmoid](/assets/img/deeplearning/one-stop/sigma-sigma-prime.jpg)

**tanh**

函数图和导数图

tanh‘ = 1-tanh^2

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

在梯度更新的时候需要对sigmoid（w*x+b）对w求导， => (w*x+b)' * sigmoid(f(x))' => sig(f(x))*(1-sig(f(x)))* x， 除x项外，其他都是常数，且恒大于0，从第二层起，神经元进过激活函数的输出都是正值，所以后续在做BP的时候，梯度恒为正。

**ReLU**

![relu](/assets/img/deeplearning/one-stop/relu.png)

对比sigmoid类函数主要变化是：

1）单侧抑制；

2）相对宽阔的兴奋边界；

3）稀疏激活性。

存在问题：

ReLU单元比较脆弱并且可能“死掉”，而且是不可逆的，因此导致了数据多样化的丢失。通过合理设置学习率，会降低神经元“死掉”的概率。

**Leaky ReLU**

![leaky-relu](/assets/img/deeplearning/one-stop/leaky-relu.png)

优缺点：

1. 避免ReLU神经元死亡的问题
2. 能有负值输出
3. 输入小于0时权重值是超参数

**MaxOut**
$$
Maxout(x) = max(w_1^T + b_1, w_2^T + b_2)
$$

**特殊的SoftMax**

_softmax一般只作为多分类的模型的最后一层的激活函数_，一般多分类模型的最后一个层的节点数就是类别数，把最后一层的输出总和归一，其中最大的节点值对应的类别就是预测的类别

![softmax](/assets/img/deeplearning/one-stop/softmax.jpg)

原理很简单，重点在于softmax如何进行梯度更新

损失函数一般采用交叉熵

$$
Loss = -\sum_iy_ilna_i
$$
在多分类问题中，N长的label向量其中只有1个1其他全都为0，所以交叉熵中的求和可以忽略掉

![softmax-hand1](/assets/img/deeplearning/one-stop/softmax-hand1.jpg)

![WeChat Image_20190819163357](/assets/img/deeplearning/one-stop/softmax-hand2.jpg)

上面两个图片中有一个问题，为什么节点4要对w5i w6i 求偏导，他们又没有直接连接，因为softmax把节点456连接了起来，所以需要考虑节点4对56的影响；不然网络一次只更新一部分权重

从上面的推导可以看出，对于label为1节点相连的权重求偏导的时候g = (a-1)*o; 对于其他权重g = a*o

**Reference**

[常见激活函数的比较](https://zhuanlan.zhihu.com/p/32610035)

[softmax](https://zhuanlan.zhihu.com/p/25723112)



### 梯度爆炸/消失

梯度爆炸和梯度消失在某种程度上其实是一回事，在BP过程中，如果激活函数的梯度大于1，那么经过链式求导法则之后，梯度值会以指数形式增加，发生梯度爆炸，反之会以指数形式缩小，发生梯度消失。

解决梯度消失的两个办法

1）、使用 ReLU、LReLU、ELU、maxout 等激活函数

sigmoid函数的梯度随着x的增大或减小和消失，而ReLU不会。

2）、使用批规范化

通过规范化操作将输出信号x规范化到均值为0，方差为1保证网络的稳定性。从上述分析分可以看到，反向传播式子中有w的存在，所以w的大小影响了梯度的消失和爆炸，Batch Normalization 就是通过对每一层的输出规范为均值和方差一致的方法，消除了w带来的放大缩小的影响，进而解决梯度消失和爆炸的问题。



### 为什么神经网络中会使用交叉熵作为损失函数？

**特别注意，这里是说最后输出层的损失函数和激活函数**

***输出层的激活函数，和hidden layer的激活函数是分开设置的***

要说为什么采用交叉熵，就要先看看为什么不用均方误差

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
Reference

http://heloowird.com/2017/03/08/diff_errors_of_neural_network/

### 输出层的损失函数和激活函数选择

二分类： sigmoid+交叉熵

多分类：softmax+交叉熵

回归：线性激活函数+均方误差

**输出层的激活函数是按照任务来定的，二分类就是sigmoid，多分类是softmax，回归是线性激活函数**，但是在hidden layer中，为了抑制梯度消失，一般采用Relu。当sigmoid/softmax作为最后一层激活函数的时候为了让最后一层也可以加速梯度更新，抑制梯度消失，一般使用交叉熵作为损失函数。

### Batch Normalization

#### 为什么想要去做BN？

神经网络，特别是深度神经网络，随着梯度更新，各层权重w的值会增大，深层的网络激活函数输出已经到了饱和区（使用饱和激活函数例如sigmoid），浅层的梯度更新缓慢；称为Internal Covariate Shift(ICS)问题

解决梯度饱和问题的思路有两种

1：更换激活函数-Relu，Leaky Relu

2：从激活函数的输入分布入手，修正输入分布，使之落在饱和激活函数的敏感区（0值区附近）

BN层就是从第二个角度出发的。

#### 算法思路

首先要说的是Batch Normalization是基于Mini Batch的，并且作用于激活函数之前

对于每一个mini-batch的中的数据的每一个特征，基于特征粒度的z-score normalization，除数+一个基数是为了避免方差为0；

$$
\hat{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$
这样处理之后，激活函数的输入是均值为0，方差为1的N个特征值；缓解了梯度饱和的问题，使得输入落在激活函数的敏感区间，**可是这样带来了一个更大的问题，经过normalization后的数据的表达能力下降**，为什么？因为均值为0，方差为1 的输入会落在sigmoid函数的0值附近，**进入了非线性激活函数的线性区域，丧失了非线性的表达能力**

因此，BN又引入了两个**可学习（learnable）的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta)** 。这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BZ_j%7D%3D%5Cgamma_j+%5Chat%7BZ%7D_j%2B%5Cbeta_j) 。特别地，当 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%5E2%3D%5Csigma%5E2%2C%5Cbeta%3D%5Cmu) 时，可以实现等价变换（identity transform）并且保留了原始输入特征的分布信息。除非完全等价还原，否则一定损失了部分表达能力。用数据的表达能力换取模型的更快拟合。

整个过程就是通过normalization把各式各样的分布拉回到标准正态分布，设置一个全局的base，底色，均值为0，方差为1的标准正态分布，然后通过反拉伸和反偏移，把数据分布往原来的方向推回一部分，在表达能力和拟合速度之间取得一个平衡

**预测时候BN层的使用**

预测的时候没有办法计算预测样本的均值和方差，就**采用所有训练样本的均值和方差**，这是对样本分布的真实均值和方差的无偏估计

**BN层总结：**

1. **使得每一层的输入数据的分布相对稳定，加快模型的拟合速度**，深层不需要适应浅层的权重变化，层与层之间解耦合。每一层单独学习
2. **使得模型对参数不敏感**，由于BN层的加入，输入值相对稳定，上一层权重的比例缩放变化在下一层没有体现。避免浅层网络参数的变化蝴蝶效应般的影响到后面的深层网络
3. **BN层允许模型使用饱和激活函数，缓解梯度消失问题**
4. **BN层有一定的正则化效果，抑制过拟合**；对于任意一个样本，和不同的其他样本组合成mini-batch，它自己变化后的值都是不同的，相当于给模型添加了噪声，抑制过拟合

Reference

https://zhuanlan.zhihu.com/p/25723112

https://zhuanlan.zhihu.com/p/34879333

## Time Series Model

### RNN

RNN能给传统的神经网络带来了时间维度的考虑，传统的DNN 假设输入之间是完全独立的。假定现在有一个分类的需求，一个球需要判断是往右滚还是往左滚，输入的label是球的坐标，在不给定时间序列的情况下，球左右滚都有可能，但是一定给定时间上的先后顺序，那么球的滚动方向就确定下来了。另外人类的语音输入也是具有强烈的前后关联的，单独把每个词分开看，并不能或者很难看出语句的含义。

RNN则是在前馈神经网络上添加一个传递先前信息的循环，把上一个输入的hidden state 传递给下一个状态，但是如果状态数过多，太早的状态的信息由于梯度消失问题而消失，所以RNN只具有短期记忆

### LSTM

谈到LSTM之前首先要说***Recurrent Neural Network**

## NLP

### TF-IDF

TF(term frequency) = 词在文章中出现的次数/文章的总词数

IDF(Inverse Document Frequency) = log(语料库总文章数/包含该词的文章数+1)

TF-IDF = TF*IDF

于是一个词的TF-IDF和词出现次数正比，和词出现的文章数反比，因为如果一个词在较少的文章中出现，那么这个词就更有区分度

注意TF计算是针对这一篇文章，而IDF是针对整个语料库
