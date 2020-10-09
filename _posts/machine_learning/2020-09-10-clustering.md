---
layout: post
title: 聚类算法模型汇总
category: ML
tags: Machine Learning
description: 
---

<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
</head>

## 聚类

### 是否适合聚类

聚类不同于分类算法有一个最优化目标，聚类是一种统计方法，把相似的数据聚在一起。 在进行聚类之前，有必要对数据有一个初步的理解，如果数据是纯随机分布的话，虽然任何一个聚类算法都可以“强行”得到一个结果，但是这个聚类效果是没有意义的；常用**Hopkins Statistic** 霍普金斯统计量，来判断，越接近1，聚类越有意义.

1. 首先从所有样本中随机选择n个点,记为$ p_1,p_2,p_3...p_ n $,对每一个点找到离他最近的样本点，并且计算距离，记为$x_1,x_2,x_3...x_n$
2. 在样本可能的取值范围内随机生成n个点记为$q_1,q_2,q_3...q_n$,找到离他最近的样本点，计算距离记为$y_1,y_2...y_n$
3. 霍普金斯统计量可以表示成$ H = \frac{\sum_{i=1}^{n}y_i}{\sum_{i=1}^{n}y_i +\sum_{i=1}^{n}x_i} $，如果样本纯随机分布，那么x和y的求和值应该很接近，反之y之和应该大于x之和，越接近1，说明样本越可能有非随机分布，越接近0.5说明样本随机分布。

**Reference**<br>[python hopkins statistic 实现](https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/)

### K-means

K-means涉及到向量距离的衡量，一般采用欧氏距离

**算法步骤：**

1.为每个聚类确定一个初始聚类中心（一般就是选取K个样本点），这样就有K 个初始聚类中心。

2.将样本集中的样本按照最小距离原则分配到最邻近聚类

3.使用每个聚类中的样本均值作为新的聚类中心。

4.重复步骤2.3直到聚类中心不再变化，或者变化很小

5.结束，得到K个聚类

准则函数SSE(within-cluster sum of squared errors)

$$
E = \sum_{i=1}^{k}\sum_{X\in{C_i}}{|X-\bar{X_i}|}^2
$$

来评价聚类性能,让SSE最小

特点：

1. 简单，快捷，时间复杂度O (nkt ) , 其中, n 是所有对象的数目, k 是簇的数目, t 是迭代的次数。通常k < <n 且t < <n
2. 当结果簇是密集的，而簇与簇之间区别明显时, 它的效果较好
3. 对**初始聚类中心敏感，对初始k敏感，对噪声敏感**
4. 对于**非凸数据集效果不好**
5. 由于k-means是基于距离计算相似度，如果特征的量纲不同，结果很容易被高数据值的特征支配，**所以必须进行正则化z-score 正则， 同时，可以利用这一特点，给某些特征增大权重，只须在正则化之后乘上权重即可。**

改进：

k-中心点算法：k -means算法对于孤立点是敏感的。为了解决这个问题，不采用簇中的平均值作为参照点，可以选用簇中位置最中心的对象，即中心点作为参照点。这样划分方法仍然是基于最小化所有对象与其参照点之间的相异度之和的原则来执行的。

K-means++ 算法：

选择一个样本点作为第一个初始点，计算所有其他样本点距离现在已有的聚类中心中最近的哪一个的距离，选择全部距离中最大的那个点作为新的聚类中心，直到选出K个聚类中心，然后初始化这K个中心，按照传统K-means进行聚类

**怎么确定K值？**

1. 基于先验知识
2. 通过从小到大选取不同的K值，画出SSE曲线，找到那个肘型点，即SSE数值降低突然放缓的那个点
3. 基于轮廓系数1，计算样本i到同簇其他样本的平均距离ai。ai 越小，说明样本i越应该被聚类到该簇。将ai 称为样本i的**簇内不相似度**，**簇C中所有样本的a i 均值称为簇C的簇不相似度。**2，计算样本i到其他某簇Cj 的所有样本的平均距离bij，称为样本i与簇Cj 的不相似度。定义为样本i的**簇间不相似度**：bi =min{bi1, bi2, ..., bik}**bi越大，说明样本i越不属于其他簇。**3，根据样本i的簇内不相似度a i 和簇间不相似度b i ，定义样本i的**轮廓系数**：

4，判断：

si接近1，则说明样本i聚类合理；此时a接近0

si接近-1，则说明样本i更应该分类到另外的簇；此时b接近0

若si 近似为0，则说明样本i在两个簇的边界上。

计算所有i的轮廓系数，求出平均值即为当前聚类的整体轮廓系数，度量数据聚类的紧密程度

从小到大遍历K，在每个k值上重复运行数次kmeans(避免局部最优解)，并计算当前k的平均轮廓系数，最后选取轮廓系数最大的值对应的k作为最终的集群数目。

### DBSCAN

直接从图中来解释一些名词：

ϵ邻域： 距离对象ϵ距离之内的区域；对应图上圆圈

核心对象：ϵ邻域内至少有MinPts个其他对象的对象；对应图上红色圆圈

密度直达：如果p在q的邻域中，且q是核心对象，则对象p从q出发是密度直达的；例如A红色圆圈中的点从A都是密度直达，但是从对象到A不一定成立，因为对象不一定是核心对象

密度可达：两个核心对象x,y 通过密度直达链可以相连则称x,y密度可达；图中箭头相连的点都是密度可达的，密度可达也不一定是相互的，对应图中单箭头

密度相连：两个非核心对象，通过密度直达到核心对象，然后核心对象之间密度可达，最后两个非核心对象建立连接，就是密度相连；对应图中所有所有红色圆圈覆盖区域内的点都是密度相连的

算法思想：

DBSCAN的聚类定义很简单：由密度可达关系导出的最大密度相连的样本集合，即为我们最终聚类的一个类别，或者说一个簇。

这个DBSCAN的簇里面可以有一个或者多个核心对象。如果只有一个核心对象，则簇里其他的非核心对象样本都在这个核心对象的ϵϵ-邻域里；如果有多个核心对象，则簇里的任意一个核心对象的ϵϵ-邻域中一定有一个其他的核心对象，否则这两个核心对象无法密度可达。这些核心对象的ϵϵ-邻域里所有的样本的集合组成的一个DBSCAN聚类簇。

那么怎么才能找到这样的簇样本集合呢？DBSCAN使用的方法很简单，它任意选择一个没有类别的核心对象作为种子，然后找到所有这个核心对象能够密度可达的样本集合，即为一个聚类簇。接着继续选择另一个没有类别的核心对象去寻找密度可达的样本集合，这样就得到另一个聚类簇。一直运行到所有核心对象都有类别为止。

**那剩下的非核心对象而且不属于任何一个聚类簇的怎么办？**

1.  outlier不在任何一个核心对象的邻域内，就会被标记成噪声
2.  如果一个点在两个核心对象的邻域内，但是自己不是核心对象，遵守先来后到的原则划分聚类簇

**适用性和优缺点：**

- 适用于样本密度大，且非凸的情况
- 在聚类的同时发现outlier，对异常值不敏感
- 初始化不影响总体结果，但是也不是完全稳定的，存在先来后到的情况，不同的选取顺序也会有影响
- 密度不均匀，类内间距大的时候不适用
- 于k-means相比对了一个参数需要调参，邻域范围和最小样本阈值都需要调参，且影响很大

### GMM

 Gaussian mixture model

### 效果评估

- 紧密性(compactness)类点到聚类中心的平均距离 $\overline{CP} = \frac{1}{K}\sum_{k=1}^{K}\overline{CP_k}$
- 间隔性(Separation) 各聚类中心两两之间的平均距离

$$
\overline{SP}=\frac{2}{k^2-k}\sum_{i=1}^{k}\sum_{j=i+1}^{k}|w_i-w_j|
$$

- 戴维森堡丁指数(Davies-Bouldin Index) 任意两类的类内平均距离之和除以两聚类中心距离的最大值的平均，越小表示类内距离小，类间距离大。固定一个类，和其他类去计算上面的值，取最大，然后固定另一个类，重复，最后取平均

$$
DB = \frac{1}{k}\sum_{i=1}^{k}max(\frac{\overline{C_i}+\overline{C_j}}{|w_i-w_j|})
$$

- 邓恩指数(Dunn Validity Index, DVI) 任意两个簇类元素的最短距离(类间)除以任意簇类中的最大距离(类内)。DVI越大说明类间距离越大，同时类内距离越小
- 如果数据有标签，可以简单的使用**Cluster Accuracy(CA)**，正确聚类的数据数目占总数据数目的比例。值得一提的是，**聚类并不是意味着一定没有标签**，例如我们有大量没有标签的数据需要做聚类，但是我们只有一小部分知道标签的数据，可以用这部分已知标签的数据去确定聚类算法的参数值，或者比较不同算法的好坏，或者找出那些特征更加适合做聚类
- 兰德系数(Rand Index, RI),类似于分类,其中TP表示是一簇的数据被分为同一簇，FP不应该聚在一起的被聚在一起，TN不应该聚在一起的被正确分开，FN应该分开的被聚在一起。

$$
RI = \frac{TP+FP}{TP+TN+FP+FN}
$$

- F-Score 类似于分类聚类也有F-score的概念

$$
P = \frac{TP}{TP+FP} \quad 精确率\\
R = \frac{TP}{TP+FN} \quad 召回率\\
F_\beta = \frac{(1+\beta^2)PR}{\beta^2P+R}
$$

### 模型之间的异同

**K-Means 和 DBSCAN的区别**

**1)K均值和DBSCAN都是将每个对象指派到单个簇的划分聚类算法，但是K均值一般聚类所有对象，而DBSCAN丢弃被它识别为噪声的对象。**

**2)K均值使用簇的基于原型的概念，而DBSCAN使用基于密度的概念。**

**3)K均值很难处理非球形的簇和不同大小的簇。DBSCAN可以处理不同大小或形状的簇，并且不太受噪声和离群点的影响。当簇具有很不相同的密度时，两种算法的性能都很差。**

4)K均值只能用于具有明确定义的质心（比如均值或中位数）的数据。DBSCAN要求密度定义（基于传统的欧几里得密度概念）对于数据是有意义的。

5)K均值可以用于稀疏的高维数据，如文档数据。DBSCAN通常在这类数据上的性能很差，因为对于高维数据，传统的欧几里得密度定义不能很好处理它们。

6)K均值和DBSCAN的最初版本都是针对欧几里得数据设计的，但是它们都被扩展，以便处理其他类型的数据。

7)基本K均值算法等价于一种统计聚类方法（混合模型），假定所有的簇都来自球形高斯分布，具有不同的均值，但具有相同的协方差矩阵。DBSCAN不对数据的分布做任何假定。

8)K均值DBSCAN和都寻找使用所有属性的簇，即它们都不寻找可能只涉及某个属性子集的簇。

**9)K均值可以发现不是明显分离的簇，即便簇有重叠也可以发现，但是DBSCAN会合并有重叠的簇。**

**10)K均值算法的时间复杂度是O(m)，而DBSCAN的时间复杂度是O(m^2)，除非用于诸如低维欧几里得数据这样的特殊情况。（k-means只需要计算每个点到K个中心的距离，DBSCAN计算每个点到其他点的距离）**

**11)DBSCAN多次运行产生相同的结果，而K均值通常使用随机初始化质心，不会产生相同的结果。**

**12)DBSCAN自动地确定簇个数，对于K均值，簇个数需要作为参数指定。然而，DBSCAN必须指定另外两个参数：Eps（邻域半径）和MinPts（最少点数）。**

13)K均值聚类可以看作优化问题，即最小化每个点到最近质心的误差平方和，并且可以看作一种统计聚类（混合模型）的特例。DBSCAN不基于任何形式化模型。