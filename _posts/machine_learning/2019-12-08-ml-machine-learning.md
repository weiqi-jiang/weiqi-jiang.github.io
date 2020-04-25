---
layout: post
title: 一站式 Machine Learning
category: ML
tags: Machine Learning
description: machine learning
---

## 背景知识

此部分主要是一些通用的背景知识，不局限于机器学习领域，在其他领域也有广泛应用。

### 频率派和贝叶斯派以及贝叶斯公式

#### 频率学派

频率学派相信一个概率事件发生的概率是存在唯一的True Value的，客观存在，只是人们不知道，我们想要知道这个真值就要用大量的重复独立实验去逼近，根据大数定理，在实验次数足够大的情况下，事件发生的频率无限逼近概率

#### 贝叶斯学派

贝叶斯学派认为事件发生的概率不是一个确定的值，而是一个分布，

A指参数，P(A)称为先验概率，表示参数取某个特定值的概率，这个是利用其它先验知识获得的。如果先验函数是均匀分布，那么贝叶斯方法等于频率方法，先验为均匀分布也就是所有参数情况的概率一样，也就是没有先验，先验的作用就是在于能提前给定一些信息，有偏向。

P(B|A)表示似然函数，意思是在A为某个值的情况下，发生某个特定事件B的概率。

P(B) 是发生某个事件B的概率，他是一个累计的结果，是全概率公式计算所得 = 在参数A1情况下发生B的概率*参数A1的概率 + 事件A2情况下发生B的概率\*参数A2的概率 + ... 在参数An发生的情况下发生B的概率\*参数An的概率

P(A|B)为后验概率， 发生某个事件的情况下，参数为A的概率。那么实际使用中，例如贝叶斯分类器，通常取用使得后验概率最大的参数

![bayes](/assets/img/ML/one-stop-machine-learning/bayes.jpg)

贝叶斯学派和统计学派的区别可以看成概率和统计的区别，概率是已知模型和参数推测结果，统计是已知结果倒推出模型和参数。

#### 贝叶斯公式的理解

以一个例子来理解贝叶斯公式，假设事件B表示车辆警报响了，事件A表示车辆被砸，P(A|B)表示警报响了的情况下车子被砸的概率，P(B|A)似然函数表示车子被砸了警报响的概率，姑且认为是1. P(A)先验概率表示车辆被砸的概率，利用先验知识知道这个值很小。P(B)表示警报响的概率，需要考虑很多情况，由车辆被砸导致的警报响，还是其他原因导致的警报响。

有两个理解点：

1. 如果P(B)越大，也就是说警报概率越大，越常见，那么在车辆被砸概率一定的情况下，P(A|B)越小，反之如果警报概率小，而且只要车辆被砸就会引发警报，那么在车辆被砸概率一定的情况下，后验概率大。也就是说考虑到了警报响这件事本身的概率，换句话说考虑到了非车辆被砸情况导致的警报响。
2. 考虑先验，我们提前知道P(A)很小，分子很小， P(A|B) 在其他已知的情况下，不会很大。先验知识通常由已知数据统计而来，数据量越大，先验概率越准

#### 最大似然估计(MLE)

似然函数P(B|A) 表示在不同参数下，事件B发生的概率，现在事件B已经发生，那么似然函数最大对应的那个参数值最有可能导致事件B的发生，我们更有信心去相信模型的参数是似然函数最大对应的参数。

拿硬币的来举例，如果现在事件是 ‘反正正正正反正正正反’。硬币正向向上的概率为θ，那么似然函数是

f(x0,θ)=(1−θ)×θ×θ×θ×θ×(1−θ)×θ×θ×θ×(1−θ)=θ^7(1−θ)^3=f(θ) , 此时如果画出似然函数的图像可以看出θ为0.7左右的时候似然函数概率最大，在该事件发生的情况下，用最大似然估计的方法估计出的正面向上的概率值是0.7。但是这仅仅根据一次投掷结果就推测实在有点武断，需要引入先验概率，也就是最大后验概率估计。

![](C:\Users\Jiang\Documents\GitHub\weiqi-jiang.github.io\assets\img\ML\one-stop-machine-learning\MLE.png)

#### 最大后验概率估计(MAP)

最大后验概率，就是最大化似然函数和先验函数的乘积(分母事件已经发生，边缘概率已知为定值)，还是拿上面那个例子来举例，已经发生的事件是 ‘反正正正正反正正正反’，似然函数是f(θ) ，我们根据先验知识，先验的认为P(θ) 满足一个均值为0.5，方差很小的正态分布，计算P(B|θ)*P(θ)，其实就是对之前的似然函数，用先验函数进行修正，此时后验函数最大值向0.5偏移。

那么如果我们加大实验次数呢？似然函数会越来越收窄，此时先验函数对似然函数的“修正作用”越来越小，最大后验概率估计结果接近最大似然估计。

**MLE 和 MAP的区别联系**

MLE 可以看成把先验函数视为均匀分布的MAP, MAP在实验次数最够多的情况下结果和MLE一样。



**Reference**

[详解MLE,MAP以及贝叶斯公式的理解](https://blog.csdn.net/u011508640/article/details/72815981)

[频率学派还是贝叶斯学派](https://www.sohu.com/a/215176689_610300)



### 距离的衡量

作为距离测度满足几个条件：

1. 非负性 d(x, y)>=0
2. 自反性d(x, y) = 0 当且仅当 x  ==  y
3. 对称性d(x, y) = d(y, x)
4. 三角不等式 d(x, y) < d(x, z) + d(z, y)

**闵科夫斯基距离**
$$
L_p(x_i,x_j) = (\sum_{l=1}^{n}|x_i^l - x_j^l|^p)
$$

上面是闵科夫斯基距离公式，可以看到p=2时称为欧式距离；当p=1时，称为曼哈顿距离；当p为无穷大时，是各个坐标距离的最大值，称为切比雪夫距离。

**余弦距离**

![cosine-distance](/assets/img/ML/one-stop-machine-learning/cosine-distance.png)

**皮尔森相关系数**

两者的协方差/标准差的乘积

![pearson-correlation](/assets/img/ML/one-stop-machine-learning/pearson-correlation.png)

**对于标量特征**

主要是Lp距离，皮尔森距离等**特别注意的是为了避免数值大的特征支配距离衡量，需要进行数值归一化**

**对于二元特征**

一般采用**取值不同的同位属性数/单个元素的属性位数，**来衡量距离；上面所说的相异度应该叫做**对称二元相异度**。现实中还有一种情况，就是我们只关心两者都取1的情况，而认为两者都取0的属性并不意味着两者更相似。例如在根据病情对病人聚类时，如果两个人都患有肺癌，我们认为两个人增强了相似度，但如果两个人都没患肺癌，并不觉得这加强了两人的相似性，在这种情况下，改用“**取值不同的同位属性数/(单个元素的属性位数-同取0的位数)**”来标识相异度，这叫做非对称二元相异度。如果用1减去非对称二元相异度，则得到**非对称二元相似度**，也叫**Jaccard系数**，是一个非常重要的概念。

**对于类别特征**

可以先one-hot再采用二元特征方式衡量，也可以用”**取值不同的同位属性数/单个元素的全部属性数**”衡量。

**对于序数变量**

序数变量是具有序数意义的分类变量，通常可以按照一定顺序意义排列，如冠军、亚军和季军。对于序数变量，一般为每个值分配一个数，叫做这个值的秩，然后以秩代替原值当做标量属性计算相异度。

**对于向量**

由于向量有方向，一般采用cosine距离衡量，**cosine衡量的是相似性而不是相异性**



**Reference**

[常见的距离测度](https://blog.mythsman.com/post/5d2d440da2005d74040ef6e8/)



## 基础知识

此部分主要涉及机器学习相关的通用基础知识，相较于背景知识，此部分更加贴近机器学习领域，但是大多是不同模型之间通用的知识，比较泛化。其中梯度和激活函数的部分，虽然在传统机器学习领域也有涉及，但是在深度网络中才更加的大放异彩，故放在深度学习的基础知识部分进行记录。

### 机器学习的本质

> 机器学习的本质就是对问题真实模型的逼近

我们希望知道我们选择的假设和问题的真实解之间的差距，假设解和真实解之间的差距就是***风险*** ， 风险的真实大小我们无从得知，我们只能从测试样本的模型预测结果和真实结果之间的差距去表示，因为测试样本是标记过的样本，真实值已知，这个测试集和真实值的差距就是**经验风险**， 之前的机器学习方法的优化目标是**经验风险最小化原则**，例如极大似然估计，当模型是条件概率分布，损失函数是对数函数的时候，经验风险最小化就是极大似然估计

后来发现只要足够复杂的模型，去记住样本集中每一个样本的特征，就能够在样本集上完成100%分类正确，但是在样本集以外错误率很高，overfitting。经验风险最小化原则训练出的模型能描述真实问题的前提是经验风险和真实风险能够无限逼近，具有一致性。可是样本集的分布是不是代表真实样本分布，这要画上一个问号，训练模型在此样本集的基础上做到了经验风险最小，真实样本中当然会出现误差变大的情况。

因此，真实风险不能只用经验误差去描述，应该有两部分组成：**经验风险，模型在样本上的误差；置信风险，代表多大程度上相信模型在未知样本上的结果，**此时的误差叫做***泛化误差界***

置信风险我们无法精确衡量，我们只能估计一个大概的置信区间，使得整个误差只能计算上界，因此叫做泛化误差界。

置信风险和两个量有关，样本集的数量n，n越大，我们越有理由相信模型在未来真实表现更好；另一个是模型的VC维h，VC维越高，模型越复杂，越容易过拟合，于是经验风险最小化原则，变成了**结构风险最小化原则**

**泛化误差界**：R(w)(真实风险)≤Remp(w)（经验风险）+Ф(n/h)（置信风险）

![error-generaliztion-bound](/assets/img/ML/one-stop-machine-learning/Error-generalization-bound.png)

N代表样本数量，d代表假设空间的数量，VC维越高，模型 越复杂，参数越多，假设空间大小越大

![structure-error](/assets/img/ML/one-stop-machine-learning/structure-risk.png)

在统计学习方法中，结构风险等于经验风险加上惩罚项，当模型是条件概率分布，损失函数是对数损失函数，复杂度由模型先验概率表示时，结构风险最小化就是最大后验概率估计



### One-Hot 编码

#### 为什么使用one-hot编码？

1. 使得模型可以解决类别问题，计算机只能处理数值，不能处理属性，例如‘蓝色’，需要把属性硬编码成数字，然后对编码结果进行one-hot
2. 距离计算合理，属性硬编码之后，数值之间的差异不代表距离的大小，只代表不同的属性取值，one-hot之后距离差相同，更加合理

#### Python常用编码API

pandas pd.get_dummies()

- 同时适用于字符型和数字型的数据
- 因为没有onehot-encoder的‘记忆功能’，如果不能同时对测试集和训练集做one-hot的话，会出现维度错误

skilearn OneHotEncoder()

- 只适用于数值型数据
- 如果训练集足够大，出现了所有可能取值，那么transform 函数可以对未知测试集适用



### Bias-Variance Trade-Off

bias 和variance 还有随机误差是训练模型和真实模型之间误差的三个组成部分。随机误差这个无法避免，暂不讨论。

bias：“用所有可能的训练数据集训练出的所有模型的输出预测结果的期望”与“真实模型”的输出值（样本真实结果）之间的差异，也就是在衡量对样本的拟合效果，拟合的越好，bias越小，但是越容易过拟合，一般增加模型复杂度，有助于降低bias

variance：是“不同的训练数据集训练出的模型”的输出值之间的差异，也就说对于随机样本预测误差的波动大小，约束模型复杂度有助于降低variance。

很明显，variance和bias之间存在一个取舍，加大模型复杂度，bias会变小，但是同时模型容易过拟合，variance变大，反之亦然。

![bias-variance-tradeoff](/assets/img/ML/one-stop-machine-learning/variance-bias-tradeoff.jpg)

bias-variance曲线大概是这样的一个样子，当total 在拐点右边时，模型过拟合，左边时，模型欠拟合。

欠拟合的标志：

1. 训练集误差大
2. 验证集和测试集误差差不多大

过拟合标志：

1. 训练集误差小
2. 测试集误差大

 **Reference**

[方差偏差均衡](https://plushunter.github.io/2017/04/19/机器学习算法系列（18）：方差偏差权衡（Bias-Variance Tradeoff）/)



### ROC &AUC

要谈到ROC图和AUC的值，首先要提到混淆矩阵

![confusion-matrix](/assets/img/ML/one-stop-machine-learning/confusion-matrix.png)

ROC曲线**横坐标FPR**，**纵坐标TPR，**样本中的真实正例类别总数即TP+FN。`TPR`即True Positive Rate，TPR = TP/(TP+FN)。
同理，样本中的真实反例类别总数为FP+TN。`FPR`即False Positive Rate，FPR=FP/(TN+FP)

接下来我们考虑ROC曲线图中的四个点和一条线。第一个点，**(0,1)（左上角，完美）**，即FPR=0, TPR=1，这意味着FN（false negative）=0，并且FP（false positive）=0。Wow，这是一个完美的分类器，它将所有的样本都正确分类。第二个点，**(1,0)（右下角，最差）**，即FPR=1，TPR=0，类似地分析可以发现这是一个最糟糕的分类器，因为它成功避开了所有的正确答案。第三个点，**（左下角，把所有样本分为负样本）**，即FPR=TPR=0，即FP（false positive）=TP（true positive）=0，可以发现该分类器预测所有的样本都为负样本（negative）。类似的，第四个点**（1,1）（右上角，把所有样本分为正样本）**，分类器实际上预测所有的样本都为正样本。经过以上的分析，我们可以断言，ROC曲线越接近左上角，该分类器的性能越好

下面考虑ROC曲线图中的虚线y=x上的点。这条对角线上的点其实表示的是一个采用随机猜测策略的分类器的结果，例如(0.5,0.5)，表示该分类器随机对于一半的样本猜测其为正样本，另外一半的样本为负样本。

**那ROC曲线是怎么画出来的呢？**

假设有N个样本，对应的概率值p（取值0-1）分别为p1，p2.... pn

分别把pi作为threshold 大于threshold的判别为1，小于的判别为0，就可以得到N个ROC曲线上的点，当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。

**AUC的值就是曲线下的面积（area under curve）AUC值是一个概率值，当你随机挑选一个正样本以及一个负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值**。

ROC曲线和AUC的值有一个很好的特性就是当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变；这个在验证模型效果的时候很好用，做k-fold cv的时候验证集正负样本分布式变化的，如果衡量指标有较大的变化，不好衡量模型的具体效果

AUC计算方法

既然AUC描述的是正样本概率值大于负样本的概率，那么假设正样本M，负样本N，一共有M*N个正负样本对

1. 正样本预测概率大于负样本 记为1
2. 正样本预测概率等于负样本 记为0.5
3. 正样本预测概率小于负样本 记为0
4. AUC = sum/ M*N

**Reference**

[ROC & AUC](http://alexkong.net/2013/06/introduction-to-auc-and-roc/)





### L1, L2正则化

范围的作用就是把向量映射到[0, )的范围只有零向量可以取到0，用向量的范数衡量两个向量的距离

Lp-norm的定义
$$
||X||_p :=(\sum_{i=1}^{n}|x_i|^p)
$$
L1,L2的各自特点：

1. L2一定只有一条最优预测线，L1可能有多个
2. L1对异常值较之L2不敏感
3. **L1输出稀疏，把不重要的特征置位0，特征选择，L2则是保留所有特征，把特征贡献尽量压缩到最小，L1，L2共同使用就是把不重要的特征权重置为0，重要的特征的权重尽可能的低**
4. L1在0出不可导，L2处处可导，计算方便
5. L0是计算非零个数，偏向于是更多的特征权重为0，但是L0是离散值不好求解优化，L1是L0的最优凸近似，所以一般采用L1

L1,L2的函数图像和对应导数图像

![img](/assets/img/ML/one-stop-machine-learning/l1l2.jpg)

![img](/assets/img/ML/one-stop-machine-learning/l1l2-d.jpg)

Reference

https://www.zhihu.com/question/26485586



### 分类问题Metric

![WeChat Screenshot_20190811155725](/assets/img/ML/one-stop-machine-learning/classification-metric.png)

Accuracy： A = （TP+TN）/（P+N）

**当样本分布极为不均的时候，accuracy最不准确，只需要无脑预测多数类，就能有一个很好的准确率**



### 数据的归一化

有一个统一的准则，关心变量的值的模型就要归一化；只关心变量分布和变量条件概率就不需要归一化。

例如：决策树，KNN,几个算法ID3,C4.5,CART都只关系分割之后样本类别分布，概率问题，所以不要归一化

其他的优化算法都要归一化；

**z-score归一化：**

![img](https://ask.qcloudimg.com/http-save/yehe-1622570/grlnbubwqu.png?imageView2/2/w/1620)

μ，σ 分别是均值和方差

**min-max 归一化**

![img](https://ask.qcloudimg.com/http-save/yehe-1622570/pvdr0t3ygz.png?imageView2/2/w/1620)

**需要归一化的模型有**

- 神经网络，标准差归一化
- 支持向量机，标准差归一化
- 线性回归，可以用梯度下降法求解，需要标准差归一化
- PCA
- LDA
- 聚类算法基本都需要
- K近邻，线性归一化，归一到[0,1]区间内。
- 逻辑回归

**不需要归一化的模型：**

- 决策树： **每次筛选都只考虑一个变量，不考虑变量之间的相关性，所以不需要归一化**。
- 随机森林：不需要归一化，mtry为变量个数的均方根。
- 朴素贝叶斯



### 模型融合/集成学习方法

#### Bagging 和 Boosting

Bagging：从原始样本集中有放回的随机抽取若干个样本子集，用这多个样子子集分别训练多个独立的模型，最后的预测结果由多个模型表决产生。

常见Bagging模型： Random Forest

优点：

1. bagging集成和直接训练基学习器的复杂度同阶
2. bagging能不经修改的适用于多分类和回归任务
3. 使用剩下的样本可以作为验证机进行包外验证（out-of-bag estimate）

**主要关注于减少variance**

Boosting： 提升算法，模型之间是串行关系，后续模型拟合的前序模型的残差，或是拟合目标不变，但是训练样本经过加权，最后由某种方法进行线性组合生成最终的预测结果。

常见Boosting模型：AdaBoost，GBDT, XGBoost

**主要关注于减小bias**

**Bagging和Boosting的区别**：

1）样本选择上：Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。

2）样例权重：Bagging：使用均匀取样，每个样例的权重相等。Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。

3）预测函数：Bagging：所有预测函数的权重相等。Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。

4）并行计算：Bagging：各个预测函数可以并行生成。Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。

#### Stacking & Blending

stacking 用这个图就可以解释；对于每一个第一层的模型来说，假设总体1000训练集，采用5-fold，每次800训练，200预测，重复5次，得到1000 训练集完整的预测，组合N个第一层模型的1000训练集的预测，1000*N的矩阵作为第二层的训练集 ；相当于每一个第一层模型就是第二层训练集的一个“feature”，label还是原来的label

![img](/assets/img/ML/one-stop-machine-learning/stacking.png)

blending和stacking的不同就是不采用k-fold, 而是holdout，比如1000训练集，分为800训练，200验证，每个model预测这200个验证集，组成200*N的矩阵，作为下一层的训练集，也就是说stacking下一层有1000样本，而blending只有200样本

#### 总结

bagging： 引入随机性，用于减少方差，并行集成

boosting： 用于减少偏差，优化目标就是最小化残差, 串行集成

stacking： 提升预测效果

bagging 使用装袋采样来获取数据子集训练基础学习器，分类任务使用投票来融合，回归任务使用平均来融合

boosting 通过算法集合把弱分类器集成为强分类器

stacking 通过模型的堆叠提高准确度，整个元模型大致分为两层，第一层为若干个其他模型，XGBoost，RF，等等，第二层的输入为第一层的输出，一般采用LR模型把各个模型的输出进行加权融合



**Reference**

[集成学习三大法宝-bagging、boosting、stacking](https://zhuanlan.zhihu.com/p/36161812)

### 时间序列的交叉验证

1 predict second half

保证测试集发生在训练集之后，验证集发生在训练集之后，测试集之前

2 日向前链 forward-chain

![img](/assets/img/ML/one-stop-machine-learning/forward-chain.jpg)

用一天数据做验证和测试，其他时间作为训练，用一个外部循环来控制多次分割，最后平均一下测试误差

### 样本不平衡

使用正确的评估标准,当数据不平衡时可以采用percesion,recall,F1得分,MCC,AUC等评估指标。

重新采样数据集,如欠采样和过采样。**欠采样通过减少冗余类的大小来平衡数据集**。**当数据量不足时采用过采样**,尝试通过增加稀有样本的数量来平衡数据集,通过使用重复,自举,SMOTE等方法生成新的样本。

以正确的方式使用K-fold交叉验证,组合不同的重采样数据集,对多数类进行聚类。

### 防止过拟合/解决过拟合

解决方法：

1. 正则化(L1正则化,L2正则化),
2. 扩增数据集
3. 特征的筛选
4. early stop
5. dropout 按照一定的概率把hidden layer节点输出清0
6. drop connect 按照一定的概率把hidden layer的相连的权值清0
7. 集成学习方法 bagging
8. BN层

### 增长函数/对分/打散/VC维

**增长函数**

表示假设空间H，对实例m能标记的最大可能结果数

例如数据集中只有两个样本，二分类问题下，可能的结果只有4种，AA,AB,BA,BB; 此时增长函数值为4，但是可能模型或者样本有限制，使得取值达到不到4，所以增长函数的上限是2^m

**对分**

对于二分类问题来说，H中的假设对数据集D中m个实例赋予标记的每种可能结果称为对D的一种**对分（dichotomy）**。对分也是增长函数的一种上限。

**打散**

打散指的是假设空间H能实现数据集D上全部实例的对分，即**增长函数=![[公式]](https://www.zhihu.com/equation?tex=2%5Em)**

**VC维**

既然增长函数不一定能取到上限，VC维指得是假设空间H，在数据集D上能实现全打散，即增长函数可以取到上限时，数据集空间的大小，例如二维空间下，模型是线性模型，对于异或的情况不线性可分，则VC维是3，但是如果模型不是线性模型，VC维就增加，说明VC维一定程度上反映了模型的复杂度

## 数据预处理

### 特征离散化

方法：
等值分桶，等频分桶

等值分桶是指每个区间具有相同大小，但是桶中的样本量不确定；

等频分桶是指桶中样本量基本相同，但是桶的大小是变化的，需要提前做一定得数据分析，分析出数据的分布，来确定桶的上下限。

pros:

一是使的风险均摊，假设有个特征中产生一个outlier，且这个特征和label是正相关，那么如果这个outlier比average差距很大，这个特征在决定label的时候权重就会特别大，极度倾向于判别为1，相同于预测结果dominated by outlier，这是我们不愿意看到的。

二是引入非线性，提高模型的表达能力，离散化引入一定的非线性（如果模型是LR，离散化之后，one-hot之后每个特征有单独的权重，相当于引入非线性），后续one-hot特征进行交叉进一步引入非线性，离散化使得后续特征交叉更精细也更方便

三是减少计算量： 系数矩阵或者向量的点乘或内积运算速度很快

Cons：

Todo

相关思考：
是否可以把one-hot形式扩展成fuzzy logic形式？

想法： 如果换为fuzzy logic 形式计算量变大，涉及到很多浮点数的计算，大大降低计算效率，fuzzy logic 的想法在于优化离散区间；例如如果把年龄分桶，分为未成年（0-17），青年（18-25）, 壮年（26-39），中年（40-50），老年（51-） 那么如果一个人A，25岁，B，26岁他们只差一岁，实际生活中他们的兴趣爱好，经历过的事情有大致相同的趋势，代差不明显，可是在离散化之后，他们却属于不同的分组，在后续训练过程中有不同的表达。那么fuzzy logic的优势体现在它可以很好的区分年龄在不同分组的"属于"程度，体现出差异的同时，对于分桶的边界年龄两边的年龄能很大程度上体现它们的一致性。可是这种优势又不及把年龄全部展开，展开成0-100及100以上共102个区间。加上fuzzy logic 的"恐怖"的计算量，one-hot更好用



### 特征组合

pros:

在不改变输入的情况下引入非线性，解决非线性问题例如异或问题，其他的例如年龄和性别，可以组合成未成年的小男孩新特征, 这个新特征可能就和玩具枪的购买欲望成正相关。但是在没有新特征之前，购买玩具枪的欲望和年龄，性别是非线性相关的。

cons：

具体哪两个特征之间进行组合需要大量人工的测试，或者大量的先验知识。

改进：

GBDT+LR

GBDT的输入就是整个特征集，输入一个样本，GBDT 生成的M颗树，假设每棵树平均有N 个叶子节点，这N个叶子节点中只有一个输出为1（代表样本被分类到了这个叶子节点）那么最终输入到LR的特征维度有M\*N个维度，这其中只有M个特征值为1，其他为0. 这些叶子节点本身就具有了特征选择功能，如果样本被分到某个叶子节点，从root到该leaf node的path就代表一个组合特征。经过GBDT处理过的稀疏特征输入到LR中作为输入训练LR模型。LR的特征空间大小为M\*N，可以在GBDT训练时控制M,N的大小从而控制LR端的特征空间值。

### 特征选择方法

1. Filter：使用方差、Pearson相关系数、互信息等方法过滤特征，评估单个特征和结果值之间的相关程度，留下Top相关的特征部分。去掉方差小的特征，因为需要特征有分离度；

2. Wrapper：可利用“递归特征删除算法”，把特征选择看做一个特征子集搜索问题，筛选各种特征子集，用模型评估效果。

3. Embedded：可利用正则化方式选择特征，使用带惩罚项的基模型，除了选择出特征外，同时也进行了降纬。

4. 稳定性选择：采用不同的特征子集和数据子集，用模型去衡量特征的重要性，然后统计特征被认为是重要的频率。取前k个

### 特征降维方法

**PCA #todo**

**LDA #todo**

### Embedding

Embedding 的本质是降维，化稀疏为稠密（Turns positive integers(indexes) into dense vectors of fixed size）

下图是一个embedding 过程图

![062019_2051_1.png](/assets/img/ML/one-stop-machine-learning/feature-embedding.png)

其中

m: 表示样本数

feature_num ： 表示特征数

特征为稀疏的特征，不一定是one-hot形式，例如样本是文章，feature是文章相关的一些信息，比如category， tags。 tags的总个数可能有10000个，一篇文章可能有10个tag，那么tag对应的列上，该样本就有10个位为1,其他为0。 在计算的时候，不需要计算0 位乘以embedding matrix的结果，只需要计算1 位的计算值。在tensorflow中，先找到1 位的indices，计算结果相加得到    embedding之后的结果。其中embedding matrix 是需要训练的，和full-connected layer 的权重偏置一起进行训练。



## 分类/回归

### Adaboost

基本思想在于上一级分类器分错的样本会被加权对待，加权后的全体样本再次被用来训练下一个基分类器，直到收敛，收敛条件一般为达到足够小的错误率或者达到最大迭代次数

训练步骤：

1. 给每个样本赋予相同的初始权重，假设有N个样本，那么每个样本的初始权重为1/N
2. 使用全部样本训练一个基分类器，对于分类正确的样本降低权重，分类错误的样本增加权重。权重更新后的样本用于训练下一个基分类器，基分类器的的value set ->{ +1. -1}
3. 最后进行模型融合的时候，增大低误差率的分类器的权重，降低高误差率低分类器的权重

***一个基分类器的误差率\*e\*m就是被该分类器分错样本的权重总和\*，**值得注意的是**，误差率是针对单个分类器的，不是之前已经训练过的级联分类器的。**在每次训练基分类器的时候都以最小化误差率选择参数及阈值。**即误差率为模型训练的损失函数，**根据基分类器的误差率计算该基分类器在最后的复合模型中的重要程度**。**

![20141102235307399](/assets/img/ML/one-stop-machine-learning/adaboost-a.jpg)

完成一次基分类器的训练之后需要更新样本权重，更新公式如下：

![20141103000618960](/assets/img/ML/one-stop-machine-learning/adaboost-weight-update.jpg)
![WeChat Screenshot_20190626151317](/assets/img/ML/one-stop-machine-learning/adaboost-weight-update2.png)

其中：

Zm 是规范化函数，归一化函数，作用是把所有权重的和规范化为1.

Gm（x*i*）是第m个基分类器关于样本的分类结果

公式解析：

如果分类正确，Y*i* 和Gm（x*i*）结果同号，exp（-α*m*）由于 α*m*一定为正，exp（-α*m*）小于1， 那么分类正确的样本的权重就自然被压低，反正分类错误的样本的权重被提高。

最后的级联分类器由各个基分类器的线性组合完成

![20141103001155359](/assets/img/ML/one-stop-machine-learning/classifier.jpg)

**前向分布算法：**

线性模型一般如下：

![20141229215747307](/assets/img/ML/one-stop-machine-learning/forward-step.png)

其中：

b 代表基分类器

β 表示基分类器的权重

![20141229220326124](/assets/img/ML/one-stop-machine-learning/forward-step-loss.png)

L代表损失函数，线性模型的整体训练目标是最小化整体的损失函数，即达到全局最优解；悬系加法模型成为经验风险最小化的问题，即损失函数最小化的问题；简化问题的求解过程为用局部最优解逼近全局最优解，即每次只优化当前基分类器的损失函数，选择最优的参数组合使得当前单个基分类器的损失函数最小。

![20141231103543937](/assets/img/ML/one-stop-machine-learning/single-forward-loss.png)

在每次训练单个基分类器的时候，整体的优化目标如下， 由于上一级级联分类器的输出是固定的，所以只优化当前单个基分类器的参数即可，无数个局部最小化损失函数，构成了全局最小损失函数

![20141229221858912](/assets/img/ML/one-stop-machine-learning/forward-step-optimize.png)

但是为什么前向分布算法就可以无限逼近全局最优解呢？为什么样本的权重更新函数是exp函数？ 证明如下：

TODO PROOF

**在样本更新函数，以及损失函数为exp函数的时候，前向分布方法等到的最优解和直接解全局最优解的结果一样。**

[adaboost 的代码实现](https://github.com/JIANGWQ2017/ML/blob/master/adaboost/adaboost.py)

Reference

[Adaboost算法详述 ](https://zhuanlan.zhihu.com/p/42915999)，[Adaboost 算法的原理与推导](https://blog.csdn.net/v_JULY_v/article/details/40718799)



### 决策树

#### 二叉树的一些基本知识

- 高度：根节点高度为1，每一层高度加1
- 深度：根节点深度为0，每一层深度加1，可以理解为路径的个数

决策树的生成就是一个递归选择最优划分特征对训练集进行划分的过程

递归的结束条件

- 当前节点的所有sample属于同一个类别
- 当前节点sample数为零
- 当前可供划分的feature数为零

决策树的特征：

1. 过程容易理解，可视化能力强，直观
2. 应用范围广，分类回归都可以适用
3. 能够同时处理离散型和连续性的特征
4. 如果不进行剪枝操作，很容易过拟合
5. **学习一个最优的决策树是一个NPC问题，目前采用的是贪心算法，用多个局部最优去近似达到全局最优**。每次特征划分时，都是按照当前局部最优的划分特征进行划分，但是局部最优的叠加效果不一定是全局最优结果。
6. 决策树的输入不需规范化

#### ID3-C4.5-CART 算法 

**ID3**:

**支持多叉树，但是不能处理连续性变量**。采用"最大信息熵增益"作为优化目标，即当前特征的特定取值划分能达到最大信息熵增益，即以当前特征的取值划分数据，在之后的划分中，该特征不再起作用。确定划分特征之后，如果该特征有5个取值，那么该节点在划分之后产生5个child nodes，这也就是为什么ID3算法只能适用于离散型特征的原因；如果处理连续性特征，训练时，按照训练集中出现的所有可能取值划分，也可以完成训练，但是当做testing的时候，如果某一个特征中出现了训练时没有出现过得取值，那该样本应该被划分为哪一个分支就是一个问题了。“最大信息熵增益”公式如下：

![WeChat Screenshot_20190621153921](/assets/img/ML/one-stop-machine-learning/id3.png)

公式含义解析： 数据集D的经验熵即是将K类样本的信息熵相加，此时是按照类别求累计信息熵； 公式（2）特征A对数据集D的经验条件熵的含义是：按照特征A取值划分之后，每一个取值下的样本计算经验熵的总和

ID3算法的特征（优缺点）：

1： 算法偏向于取值较多的特征。如果一个特征的取值很多，每个取值对应的样本数很少，样本纯度高，倾向于使用该特征进行划分

2：没有考虑缺失值和连续性特征的问题

3：没有考虑过拟合问题

**C4.5**

在最大信息熵增益的基础上进行改进，抑制选择取值多特征的倾向，提出“最大信息熵增益比”的概念，“信息熵增益比”的公式如下

![WeChat Screenshot_20190621180220](/assets/img/ML/one-stop-machine-learning/c4.5.png)

公式解析： spilt information 中Di 的含义是特征A某个特定取值的样本集，如果特征A取值多，样本纯度高，split information的值高，那么信息增益比自然降低

增加了对连续特征的支持，对于连续性特征的处理是连续性特征二元离散化，对特征取值排序，以连续两个值中间值作为新的候选取值，假如训练集中有N个取值，那么就有N-1个候选取值，候选取值的含义是小于或者大于该值，并不仅是等于该值；

训练时，遍历所有特征，内部遍历特征下的所有可能候选取值，计算信息增益比。**选择信息增益比最大值对应的特征**(特别注意不是按照当前取值二元划分，而是按照取值所在属性，按照N-1个候选取值多元划分，所以C4.5的树是多叉树)

缺失值处理：

1：丢弃样本

2：赋予特征最常见的值

3：按照当前样本集中取值出现的概率的取值。例如A的概率0.6，B的0.4.那么缺失值有60%被分配为A，40%分配为B

#### CART（Classification And Regression Tree）

CART树是二叉树，且每个非叶子节点都有两个子节点，所以cart树的非叶子节点的数量比叶子节点数量少1。CART树即可以做回归也可以做分类，ID3,C4.5只能做分类。

CART 做回归问题的时候，采用均方误差最小化原则选择划分特征；节点预测值，是划分到该节点的样本集的label均值，计算给定划分下的均方误差总和（两边只和），遍历所有特征的所有可能取值选择均方误差最小的特征的取值进行划分。

CART做分类问题的时候，由于ID3 C4.5涉及到大量的log计算，而且C4.5还涉及到排序操作，运算复杂度高。CART算法对应改进采用gini系数，gini系数表征的是样本不纯度，故gini系数越低越好，与ID3,C4.5的metric越高越好不同。Gini系数的公式如下：图的横坐标表示概率p，二分类问题下0.5*熵和gini系数很接近

![WeChat Screenshot_20190621185537](/assets/img/ML/one-stop-machine-learning/gini.png)

公式解析：公式1 为（1 - 样本集中属于某个类别的概率平方和；）即样本集的gini指数；公式2 如果按照特征A 分割，两边的gini指数和，Di/D表示分配到两边的样本集占总样本集的比例

gini系数和熵的区别：

1：gini系数不需要复杂的log运算

2：gini偏向于连续性特征，理由同ID3偏向取值多的类似，熵偏向于离散型特征

3：熵和gini系数都是表征混乱程度，在x=1时gini系数和熵近似相等

![WeChat Screenshot_20190621200417](/assets/img/ML/one-stop-machine-learning/gini-entropy.png)

对于CART分类树连续值的处理问题，其思想和C4.5是相同的，都是将连续的特征离散化。区别在于在选择划分点时的度量方式不同，C4.5使用的是信息增益比，则CART分类树使用的是基尼系数，而且CART树只进行2分，C4.5是多分。

具体的思路如下，比如m个样本的连续特征A有m个，从小到大排列为a1,a2,...,ama1,a2,...,am,则CART算法取相邻两样本值的平均数，一共取得m-1个划分点，其中第i个划分点Ti表示为：Ti=(ai+ai+1)/2。对于这m-1个点，分别计算以该点作为二元分类点时的基尼系数。选择基尼系数最小的点作为该连续特征的二元离散分类点。比如取到的基尼系数最小的点为atat,则小于atat的值为类别1，大于atat的值为类别2，这样我们就做到了连续特征的离散化。要注意的是，与ID3或者C4.5处理离散属性不同的是，如果当前节点为连续属性，则**该属性后面还可以参与子节点的产生选择过程**。

#### Overfitting

针对过拟合问题可以通过剪枝来控制，在不剪枝之前，完全展开的决策树每个叶子节点都代表0/1 ，剪枝之后，叶子节点上划分的可能同时具有0和1样本，此时做reference的时候，如果样本被分为该节点上，则那类样本多，就认为新输入被分为哪一类。

预剪枝：

在节点划分前提前进行评估，若当前划分不能带来泛化性的提升，则停止划分，将该节点标记为叶子节点（验证方法为cross validation）

后剪枝：

先生成一颗完整的树，自下而上的评估非叶子节点，若剪枝能带来泛化性的提升，则进行剪枝操作

![081120312282_0WeChat Screenshot_20190811203030](/assets/img/ML/one-stop-machine-learning/tree-cutting.png)

剪枝过程就是最小化决策树整体的损失函数，损失函数如下，等于每一个叶子节点上样本的经验熵*样本个数求和+叶子节点个数，叶子越多，叶子节点上的经验熵越小，整体损失函数是变大还是变少，受两个方面的影响，就限制了决策树无限增殖的趋势

![WeChat Screenshot_20190811204303](/assets/img/ML/one-stop-machine-learning/tree-cutting-loss.png)

决策树代码实现：[ decision tree ](https://github.com/JIANGWQ2017/ML/blob/master/decision_tree/decision_tree.py)

**Reference**

[数据挖掘十大经典算法--CART: 分类与回归树](https://wizardforcel.gitbooks.io/dm-algo-top10/content/cart.html)

[决策树模型 ID3/C4.5/CART算法比较](https://www.cnblogs.com/wxquare/p/5379970.html)

[决策树方法小结](https://blog.csdn.net/yujianmin1990/article/details/47406037)

**Code Reference**

[机器学习之分类回归树(python实现CART)](https://www.jianshu.com/p/8863641a30b1)

[决策树思想与Python实现：CART](https://blog.csdn.net/u013719339/article/details/84502265)

[Python实现C4.5(信息增益率)](https://www.cnblogs.com/wsine/p/5180315.html)

### Random Forest

随机森林中的每一颗树都是CART树，每一颗树不进行剪枝

**Random Forest的随机性**：

randomly choose feature candidate reduce the correlation between trees

1：每个树有自己的样本集（有放回抽样得到）；

2：每个节点的在进行特征划分的时候，是从总特征集中随机抽取一个子特征集，在这个子特征集中选择最优划分特征和划分值。

优点：

1：高准确度

2：随机性的引入不容易过拟合，泛化能力强

3：处理高纬度输入，并且无需做特征选择

4：同时支持离散和连续性特征

5：训练速度快，可以得到特征importance

6：并行化计算

缺点：
1：噪声较大的分类回归问题上过拟合（为什么？）

2：树越多才稳定，但是树越多，模型越大

3：不善于处理不平衡数据集（为什么？）

### GBDT

采用的是加法模型和前向分布算法，于adaboost类似

优化函数：

![e3goBaV](/assets/img/ML/one-stop-machine-learning/gbdt-opt.png)

其中L（y, x）代表损失函数，fm-1(x) 为m-1个独立的分类器的输出,T为当前分类器的输出。损失函数可以不同，回归问题一般采用平方误差函数，分类问题一般采用指数损失函数。

**GBDT中全部都是回归树（因为只有回归树累加才有意义，分类树累加无意义）**

**每棵树拟合的都是上一个树的损失函数的负梯度方向**，对于回归树，一般采用均方误差作为损失函数，则**负梯度方向就等于残差，这里的残差具体是所有样本集的残差平均值，下一级回归树所有训练样本集的label变为残差值**

![imJx56v](/assets/img/ML/one-stop-machine-learning/gbdt-algo.png)

树与树之间是拟合上一颗树输出的残差，那么树内到底是用什么损失函数进来进行特征划分的，其实也是**平方误差损失函数，**具体的训练过程举例

![9RWud8o](/assets/img/ML/one-stop-machine-learning/gbdt-sample.png)
![NqRfj0q](/assets/img/ML/one-stop-machine-learning/gbdt-sample2.png)

**GBDT的特点：**

树的个数不能太多，后面的子树相当于在学习细节规律，如果过多，则容易造成过拟合

**GBDT和RF的适用性：**

- GBDT 对异常值很敏感，以为异常值严重影响了均方误差 ，RF不敏感
- RF减小模型variance，GBDT减小模型bias

GBDT在平方误差小于某个数的时候结束，为了防止过拟合，进行剪枝操作

### XGBoost

XGBoost 和 GBDT 在结构上几乎一样，而且都采用additive training 方法, 加法模型损失函数有一个统一的形式：

![11](/assets/img/ML/one-stop-machine-learning/xgboost-obj.png)

其中L(x,y)为任意损失函数, Ω（x）为惩罚项，f(x)为当前回归器的输出，如果损失函数为常用的平方误差函数，则目标函数如下：

![12](/assets/img/ML/one-stop-machine-learning/xgboost-obj2.png)

对该目标函数惩罚项之前的部分求梯度, 且让梯度取0， f(x) = (yi - yi-1), 当新加入的基分类器的输出正好是残差的时候，总损失函数最小，这个很好理解，我们要做的就是让基分类器的输出更可能的接近残差。

那**如果损失函数不是平方误差函数呢**？因为平方误差函数在回归问题中经常被使用，基于它的理论推导十分的成熟，有没有可能使用不同的损失函数，但是共用一套理论推导，当然有可能！

![13](/assets/img/ML/one-stop-machine-learning/xgboost-obj3.png)

对任意损失函数进行泰勒二阶展开，**为什么要二阶展开？**1是因为二阶展开往往已经对原有函数有足够高的近似程度，2是因为二阶展开具有最高二阶项，和平方误差函数类似。

- Xgboost官网上有说，当目标函数是MSE时，展开是一阶项（残差）+二阶项的形式（官网说这是一个nice form），而其他目标函数，如logloss的展开式就没有这样的形式。为了能有个统一的形式，所以采用泰勒展开来得到二阶项，这样就能把MSE推导的那套直接复用到其他自定义损失函数上。简短来说，就是为了统一损失函数求导的形式以支持自定义损失函数。这是从**为什么会想到引入泰勒二阶**的角度来说的
- 二阶信息本身就能让梯度收敛更快更准确。这一点在优化算法里的牛顿法里已经证实了。可以简单认为一阶导指引梯度方向，二阶导指引梯度方向如何变化。这是从二阶导本身的性质，也就是**为什么要用泰勒二阶展开**的角度来说的

其中要注意的事L(yi, yi-1) 是上一级分类器的损失函数，是一个固定值，省略掉常数项，损失函数有一个统一的形式

![14](/assets/img/ML/one-stop-machine-learning/xgboost-obj4.png)

后续的理论推导都是基于上式，到了只一步就可以看出XGBoost的厉害之处，上式中***包含了损失函数的一阶导和二阶导形式，但是并没有指出是哪一个具体的损失函数，也就是说任意一阶二阶可导的损失都可以适用于下述的所有推导, 使用不同的损失函数，就可以使XGBoost模型适用于不同的任务，回归，分类，排序。***

**树的复杂度(复杂度惩罚项)**

在进行后续推导之前，先要定义XGBoost的惩罚项，才方便合并表达式，方便后续推导。

**一般一棵树的复杂度由这一颗树的叶子节点个数，以及叶子节点输出值的L2范数组成**

![16](/assets/img/ML/one-stop-machine-learning/xgboost-tree-complexity.png)

**求解目标函数**

![17](/assets/img/ML/one-stop-machine-learning/xgboost-tree-complexity-loss.png)

![18](/assets/img/ML/one-stop-machine-learning/xgboost-tree-complexity-loss2.png)

![19](/assets/img/ML/one-stop-machine-learning/xgboost-tree-complexity-loss3.png)

公式解析：

- ***w函数\*负责将叶子节点映射到它对应的输出上**，*q*把输入映射到它应该在的叶子节点上
- *I* 表示在某一叶子节点上所以样本集
- 第二行的意义，前半部分是所有训练样本的在当前叶子的一阶二阶输出，注意累计i个样本，即训练样本，后本部分累加j个叶子节点的权重值。
- 理解第三行必须明白，所有叶子节点上的所有样本的个数和即为总训练样本数。所以所有训练样本的一阶二阶权重和就等于所有叶子节点上样本一阶二阶权重和

对w求导等于0，然后将w导数等于0时的值带入目标函数得：

![20](/assets/img/ML/one-stop-machine-learning/xgboost-tree-complexity-loss4.png)

此时的obj 是当前树的结构不变的情况下，仅仅改变叶子节点的权重，最极限的情况下的目标函数最小值。称为structure score

![21](/assets/img/ML/one-stop-machine-learning/xgboost-tree-complexity-loss5.png)

**训练算法**

- 枚举不同树的结构，使用上述obj作为衡量标准寻找一个最优结构的树
- 迭代进行上一步直到满足终止条件

看似简单，但是里面涉及到一个问题，如何衡量特征划分的好坏，如何确定树的结构。

决策树都采用贪心算法进行训练，每一次尝试对一个已有的叶子节点进行分割，分割的衡量增益是：

![22](/assets/img/ML/one-stop-machine-learning/xgboost-tree-gain.png)

有一个点值得注意，这个Gain表示增益当然是越大越好，可是obj应该是左子树+右子树应该小于不分割的情况啊。这是因为obj前面有负号，没有负号的Gain当然方向是反的，越大越好。

计算中不需要每次分割重新计算一遍G 和H，因为这个变量只和上一级的模型损失函数有关，不管怎么分割数值不变，所有只需要扫描一遍计算即可重复使用。但是和一般的决策树不同的是，XGBoost的训练停止条件不是无分割样本，所有样本属于同一种类，无可分割特征，**因为XGBoost加入了叶子节点复杂度惩罚项，如果分割带来的增益减去惩罚项之后增益不够大，甚至负增益那么也就不进行分割**

**xgboost 的特征重要性**

1.*importance_type=*weight（默认值），特征重要性使用特征在所有树中作为划分属性的次数。

2.*importance_type=*gain，特征重要性使用特征在作为划分属性时loss平均的降低量。

3.*importance_type=*cover，特征重要性使用特征在作为划分属性时对样本的覆盖度（就是有多少样本是通过这个特征划分开的）

**XGBoost 特点：**

- “模块化”灵活性，GBDT以CART作为基分类器，XGBoost除此之外还支持线性分类器，此时XGBoost相当于带L1,L2的线性分类器。
- 传统GBDT只利用了一阶导数信息，使用常用的平方误差损失函数，一阶导就是残差，XGBoost利用泰勒展开，使用了二阶导，实现虽然形式上和平方误差函数一致，但是更加模块化，只要函数可一阶二阶导，都可以feeds进一个理论推导框架
- 添加正则项，降低模型的variance，使得模型更加简单，避免过拟合
- 和GBDT一样有shrinkage缩减操作
- XGBoost的剪枝操作是后剪枝，先训练到最大深度，然后往前剪枝。后剪枝的好处在于不会因为一个负增益就放弃了后面可能的更大的正向增益
- 可以处理缺失值
- xgboost的并行化是特征粒度上的并行，把数据提前排序保存为block结构，在确实划分特征和划分值的时候，重复使用结构，可以并行计算

**Reference**

 [十分钟入门XGBoost(原理解析、优点介绍、参数详解）](http://www.dehong.space/XGBoost)

### Perceptron

感知机模型对应于特征空间中将实例划分为正负两类的分离超平面，故而是判别式模型

![perceptron](/assets/img/ML/one-stop-machine-learning/perceptron.png)

原始perceptron采用的激活函数是单位阶跃函数，value set {+1，-1}

由于感知机模型的输出是0和1两个离散的值，如果使用基于分类错误的平方误差，会使得损失函数不连续，更别说是否可导了。所以这里使用下面这个损失函数； 该函数在SVM模型中被称为函数间隔 margin

![perceptron-loss](/assets/img/ML/one-stop-machine-learning/perceptron-loss.png)

其中

M 表示被分类错的样本集

t 表示样本的原始类别

∅(x) 表示经过处理后的输入，w*∅(x) 表示在经过activation function之前的矩阵点乘结果  由于M 是分错类的样本集，w*∅(x) 和 t 始终异号，结果始终大于零

所以损失函数就是 |w*∅(x)| 求和，是一个连续值， 且是凸函数，凸函数可以利用梯度下降法求解，需要求解什么，就对什么求梯度。

![perceptron-gradient](/assets/img/ML/one-stop-machine-learning/perceptron-gradient.png)
![perceptron-gradient1](/assets/img/ML/one-stop-machine-learning/perceptron-gradient1.png)

由上式可以看出，下一次迭代时的权重，由上一次的权重加上学习率加权过的全部输入结果的总和（input set 是分类错的样本集），是明显的batch training，由于巨大的计算量，可以改进为随机梯度下降方法，随机取M中的一个进行梯度下降，此时的梯度下降方法跳跃很大，但是总体上是往最优值跳跃的

![perceptron-sgd](/assets/img/ML/one-stop-machine-learning/perceptron-sgd.png)

每当有分类错误点，权重更新使得分类面朝分类错误点移动

感知机收敛的条件是训练集是线性可分的，如果线性不可分，那么感知机训练过程将永远不会收敛。

感知机一旦训练到没有分类错误点就停止了，也就是即是刚刚移动到一个满足全部分类正确的位置，就停止了，没有进行最优化判断，不同的初值会影响最后的分类面。

[感知机代码实现](https://github.com/JIANGWQ2017/ML/blob/master/perceptron.py)

**Reference**

[感知机](https://www.zybuluo.com/Duanxx/note/425280)

### SVM

**SVM 是一种结构风险最小化模型**

为了理解SVM 先来了解线性模型

线性函数： g(x) = w*x + b （在空间中为hyper plane）

- 式中的x不是二维坐标系下的x轴，而是坐标向量，例如二维坐标为（x，y） = （1,2）时，x为[2,1]因为要转置和w进行点乘
- g(x) 的输出是一个连续值，在进行分类问题是，需要sign函数将连续值规范到0,1两个类别
- g(x) 在空间中是以w为法向量的一组无限个超平面，其中g(x) =0 只是其中的一个超平面，g(x) =0 这个平面称为分类面
- 将g(x)>0 的点归为1类别，g(x)<0归为-1类别，如果g（x）=0 拒绝判断

**重点来了！：**

实际中，满足条件的分类面可以不止一个，讲满足条件的一个分类面稍微“倾斜”一点，往往也可以满足要求，那么这两个同时都满足要求的分类面哪一个更好呢？这便是SVM和普通线性分类器的区别所在

**Margin**

为了区别多个可行解，确定一个解为最优解，SVM采用最大化Functional Margin

|w*x + b|表示和g(x)=0平行的平面距离g(x)=0分类面的距离，如果w*x +b 和label y同号，则分类正确，否则分类错误

于是引入函数间隔的概念：

函数间隔(Functional Margin)

![func-margin](/assets/img/ML/one-stop-machine-learning/svm-func-margin.jpg)

所有样本中函数间隔的最小值便是**数据集T关于分类面的函数间隔，**此种定义的弊端在于如果同时扩大W 和 b，在不改变超平面的情况下函数间隔扩大两倍

于是引入几何间隔的概念 geometrical margin

![geometric_margin](/assets/img/ML/one-stop-machine-learning/geometric_margin.png)

假设一个超平面经过x点且和g(x) =0 平行， x在g(x)=0上的投影为x0，几何知识可以得到 x = x0 + γ*(w/|w|)（g(x)的法向量方向），加上已知g(x0) = 0，wT * w = ||w||^2, 把前式带入后式可以得到几何间隔的表达式

几何间隔（geometrical margin）

![geometrical-margin1](/assets/img/ML/one-stop-machine-learning/geometrical-margin1.jpg)
![geometrical-margin2](/assets/img/ML/one-stop-machine-learning/geometrical-margin2.jpg)

由上两种距离的可以看出，函数间隔，用函数的值表示间隔，这是一个人工定义的量，几何间隔是空间中的真实距离,即为图中gap的一半

当分类面离数据点的“间隔”越大，分类的置信度越大，这个很好理解。分类面是两个类别的分割面，越远离这个分割面，越有confidence相信这个结果的正确

![svm-gap](/assets/img/ML/one-stop-machine-learning/svm-gap.jpg)

**求解**

设![img](https://img-blog.csdn.net/20131111154113734)γ 固定为1 ,即设样本集中离g(x)=0的分离面的距离为1，该假设不影响优化结果（为什么不影响）

优化目标 最大化：

![svm-obj-func](/assets/img/ML/one-stop-machine-learning/svm-obj-func.png)

上述优化问题等价于下式

![svm-obj-func1](/assets/img/ML/one-stop-machine-learning/svm-obj-func.jpg)

优化函数为二次函数，约束条件为线性条件，所以是一个凸二次规划问题。

并且可以利用lagrange duality 转换为对偶问题求解，对偶问题（dual problem）达到最优解的时候，原问题也同时达到最优解。

对偶问题的好处在于

- 更容易求解
- 自然的引入核函数。

利用拉格朗日对偶性以及拉格朗日乘子法可以将原问题转换为如下对偶问题，拉格朗日对偶性就是对每一个约束条件加上一个拉格朗日乘子（Lagrange multiplier) a![img](https://img-blog.csdn.net/20131111195836468)，定义拉格朗日函数（通过拉格朗日函数将约束条件融合到目标函数里去，从而只用一个函数表达式便能清楚的表达出我们的问题）

带约束条件的凸二次优化问题，利用拉格朗日乘子法转换如下

![lag-mul](/assets/img/ML/one-stop-machine-learning/lag-mul.jpg)

如果有任意一个样本点满足y(wx+b)<1， 在a无限大的情况下 ， 目标函数就会趋近于无限小。所以硬性要求所有的样本点的最小距离大于等于1.当且仅当所有样本的约束条件得到满足，即所有样本的距离都为1时，θ 等于原优化问题 1/2*（||w||）^2, 最小化该值；为了使所有约束条件满足：

1. **所有support vector 的拉格朗日乘子可以不为零，因为y(wx+b)-1对于支撑向量来说等于0**
2. **所有非支持向量的拉格朗日乘子a为0**

![lag-alpha](/assets/img/ML/one-stop-machine-learning/lag-alpha.jpg)

所以最终的优化目标函数为，且该问题和原问题等价：

![svm-equ](/assets/img/ML/one-stop-machine-learning/svm-equ.jpg)

等价问题的对偶问题为：

该问题和原问题也是等价，但解答更方便，需要3步

1. 先L(w,b,a)关于w和b最小化
2. 然后对a 求极大
3. 求解拉格朗日乘子

![1351142316_5141](/assets/img/ML/one-stop-machine-learning/lag-mul-solution.jpg)

**求解对偶问题**

第一步：

固定a，L对W和b最小化，，对w和b求偏导，并将两个偏导打入之前的L

![lag-dual1](/assets/img/ML/one-stop-machine-learning/lag-dual1.jpg)

将偏导带入到原来的L中得到：

![lag-dual2](/assets/img/ML/one-stop-machine-learning/lag-dual2.jpg)
![lag-dual43](/assets/img/ML/one-stop-machine-learning/lag-dual43.jpg)

第二步，对a求极大；公式中只有a一个变量，只要解出a，带入右边的公式就可以解除w和b

![lag-dual5](/assets/img/ML/one-stop-machine-learning/lag-dual5.jpg)
![lag-dual6](/assets/img/ML/one-stop-machine-learning/lag-dual6.jpg)
![lag-dual62](/assets/img/ML/one-stop-machine-learning/lag-dual6.png)

第三步：

SMO算法求解拉格朗日乘子，带入上面的公式得到w和b

把w带入分类面函数得到右边的形式，可以看出新样本点的输出值和训练样本的内积有关，其实只有支持向量x对应的拉格朗日乘子a有值，其他a都为0，所以计算量也不算太大

![lag-dual7](/assets/img/ML/one-stop-machine-learning/lag-dual7.jpg)

**核函数（kernal）**

使得SVM具有非线性，在线性不可分的情况下，通过核函数将所有样本映射到高维空间，在高位空间中构造出分类面，从而将线性不可分样本转换为线性可分

如果不引入核函数，构造非线性分类器的方法就是引入一个非线性映射

原分类函数：

![kernal1](/assets/img/ML/one-stop-machine-learning/kernel1.jpg)

原函数对偶形式

![kernal2](/assets/img/ML/one-stop-machine-learning/kernal2.jpg)

此时如果有一个新的映射K(x1,x2) 把<Φ(xi), Φ(x)> 映射到内积特征空间，这个函数K 就是核函数

![kernal3](/assets/img/ML/one-stop-machine-learning/kernal3.jpg)

**核函数举例**

![kernal4](/assets/img/ML/one-stop-machine-learning/kernal4.png)

假设两类样本集如左图，简单看出分类面需要是一个圆形才能将两类样本分开，样本是线性不可分的。由圆的标准方程可以得知分类面的所在方程如下

![kernal5](/assets/img/ML/one-stop-machine-learning/kernal5.jpg)

如果把二维空间上升为5维空间，五个维度分别为[X1,  X1^2,  X2,  X2^2,  X1X2], 对应的权重w 分别为[a1, a2, a3, a4, a5], 偏置b为 a6。现在原来的非线性可分的样本，在五维空间下就变成线性可分的了， g(x) = w*x + b

此时观察原函数的对偶形式(如上)，假设两个向量 x1 = (n1,n2)T, x2 = (m1,m2)T, 经过Φ(x)映射之后x1,x2变为[n1, n1^2, n2,  n2^2, n1n2], [m1, m1^2, m2,  m2^2, m1m2].

经过映射后的特征空间，内积的结果：

![kernal6](/assets/img/ML/one-stop-machine-learning/kernal6.jpg)

假设我们不进行高维映射，仅仅在原来的特征空间x1 , x2 进行变换； 观察可以得到结果和高维映射的结果很相似，但是有些许不同，这些不同可以通过缩放维度达到完全相同，如右图，缩放结果是不影响可分性的。

![kernal7](/assets/img/ML/one-stop-machine-learning/kernal7.jpg)
![kernal8](/assets/img/ML/one-stop-machine-learning/kernal8.jpg)

这两个结果看着相同，但是思想完全不同：

- 第一个是把原特征空间映射到高维空间，然后根据内积公式进行计算
- 第二个是**直接在原来的低维空间进行计算，且映射后的结果是隐式的**

此时 核函数就是:

![kernal9](/assets/img/ML/one-stop-machine-learning/kernal9.jpg)

核函数的作用就是简化了映射空间中内积计算，这个效果是很明显的，上述例子仅仅对二维空间的一阶二阶的所有组合做映射，如果原始空间为3维，所有一阶二阶三阶的所有组合有19种，那么如果原始空间的维度再往上，高维空间的维度会呈指数级爆炸，**使用核函数避开了高维空间中的计算**，并且结果等价

原问题的对偶问题

以及使用核函数后的对偶问题

![kernal10](/assets/img/ML/one-stop-machine-learning/kernal10.jpg)
![kernal11](/assets/img/ML/one-stop-machine-learning/kernal11.jpg)

σ例子中核函数很好想出来，但是实际情况下，核函数的具体形式是很难构造出来的。所以实际使用中，常常是从常用的核函数类型中选择：

- 多项式核函数 k = (<x, xi> + R)^d ， 其中两个参数<R, d>
- 高斯核函数 k = exp(- ||x1-x2||^2/(2*σ^2)), 高斯核函数可以把原始空间映射到无穷维。通过调整参数σ 高斯核函数使用十分灵活。

核函数的选取规则：

1. 如果Feature的数量很大，跟样本数量差不多，且线性可分的时候这时候选用LR或者是Linear Kernel的SVM
2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，且线性不可分的时候选用SVM+Gaussian Kernel
3. 样本数量非常多选择线性核（避免造成庞大的计算量）

4. 当样本的数量较多,特征较少时,一般手动进行特征的组合再使用SVM的线性核函数

5. 当样本维度不高且数量较少时,且不知道该用什么核函数时一般优先使用高斯核函数,因为高斯核函数为一种局部性较强的核函数,无论对于大样本还是小样本均有较好的性能且相对于多项式核函数有较少的参数

**松弛变量**

有的时候样本集对外显示线性不可分，但是如果允许若干个样本分类错误的话，往往这些分类错误的样本本身就是outlier，没有实际训练意义，样本可以转换为线性可分，或者说是近似线性可分。

![Optimal-Hyper-Plane-2](/assets/img/ML/one-stop-machine-learning/Optimal-Hyper-Plane-2.png)

图中蓝色的点就是分类错误的点，黑色线段就是分类错误带来的惩罚。

“硬间隔”分类下我们对于样本点集合距离的要求

![hard-margin](/assets/img/ML/one-stop-machine-learning/hard-margin.gif)

“软间隔”分类下引入松弛变量之后对样本点距离的要求，允许一些变量的距离小于1. ζ 即为松弛变量。可见如果松弛变量无穷大的话，所有样本点都可以满足要求，但是这样分类就变的无意义了。**使用松弛变量意味这我们放弃某些点的精确分类,**这样带来的好处也很明显，**不必使得分类面往outlier方向迁移，也可能获得更大的分类间隔**。

![soft-margin](/assets/img/ML/one-stop-machine-learning/soft-margin.gif)

可是怎么去衡量松弛变量带来的好处和坏处呢，即ζ 不能无限大啊。正则项可以选择**ζ的求和**，或是**ζ的平方和，**于是我们在原来的优化问题的基础上加上惩罚项

![clip_image002[13]](/assets/img/ML/one-stop-machine-learning/soft-margin-full.gif)

公式解析：

- 参数C描述我们有多在乎松弛项，C越大，outlier带来的影响越大，反之越小。如果C无限大，只要有一个样本不能正确分类，目标函数就会无限大，整个优化问题变成“硬间隔”分类。
- 并不是所有的点都有松弛项的，对于能够正确分类的样本，松弛量为0，对于不能正确分类的样本，才有惩罚项

**松弛变量和核函数都是用来解决线性不可分问题的，这两者有什么不同？：**

- 实际问题中，原始特征空间通常高度线性不可分，先通过核函数的形式，隐性把原始特征向量映射到高维向量，但是一般高维向量也不是完全线性可分的，是近线性可分的。、
- SVM简单来说就是**使用核函数的软间隔线性分类器**

**松弛项的改进方案：**

首先我们可以对不同的离群点采用不同的C值，对于那些绝对不能分类错的点采用很大的C值，有些不重要的点就采用很小的C值

其次，如果样本集本身的类别“偏斜”，即类别不平衡问题，一个类别样本的数量比另一个类别的样本数大的多的情况。这种情况下，minority category的样本的边界值极有可能没有达到真实边界值，分类间隔往往比理想的间隔要大，如图虚线方框是真实样本，但是在训练的时候，由于样本量有限，并么有出现该边界样本。

![image_2](/assets/img/ML/one-stop-machine-learning/relaxation.png)

这种情况下，可以对不同类别的样本采用不同的C值，少数类别样本具有更大的C值，本来就是少数类别，他们的分类正确性就应该更加重视，惩罚项优化为如下：

![clip_image002[5]](/assets/img/ML/one-stop-machine-learning/soft-margin-regulator.gif)

SVM 特点：

- 小样本
- 非线性，通过松弛变量 和 核函数 实现
- 高维模型识别
- 优秀的泛化能力，这是是因为其本身的优化目标是结构化风险最小，而不是经验风险最小，因此，通过margin的概念，得到对数据分布的结构化描述，因此减低了对数据规模和数据分布的要求
- 它是一个凸优化问题，因此局部最优解一定是全局最优解的优点
-  泛化错误率低，分类速度快，结果易解释

**缺点：**

1. 大规模训练难以实现

   SVM的空间消耗主要是存储训练样本和核矩阵，由于SVM是借助二次规划来求解支持向量，而求解二次规划将涉及m阶矩阵的计算（m为样本的个数），当m数目很大时该矩阵的存储和计算将耗费大量的机器内存和运算时间。针对以上问题的主要改进有有J.Platt的SMO算法、T.Joachims的SVM、C.J.C.Burges等的PCGC、张学工的CSVM以及O.L.Mangasarian等的SOR算法。

   如果数据量很大，SVM的训练时间就会比较长，如垃圾邮件的分类检测，没有使用SVM分类器，而是使用了简单的naive bayes分类器，或者是使用逻辑回归模型分类

2. 多分类问题存在困难

3. 对核函数和丢失数据敏感

**Reference**

[SVM原理详解](https://blog.csdn.net/abcd_d_/article/details/45094473)，

[支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)

### Linear Regression

cost Function：

![img](/assets/img/ML/one-stop-machine-learning/LR-cost-func.png)

cost function即为所有样本的均方误差

为了求解权重theta，使用梯度下降法

![img](/assets/img/ML/one-stop-machine-learning/LR-theta.png)

**Reference**

[机器学习入门：线性回归及梯度下降](https://blog.csdn.net/xiazdong/article/details/7950084)

### Logistics Regression

线性分类模型在SVM部分已经介绍了这里简单回忆下：

**线性函数：**
$$
y = w^Tx + b
$$
函数输出是连续值，用于回归问题，怎么把回归问题转换为分类问题呢。SVM采用的是把y>0 的归类为类别1，y<0的归类于类别2。LR模型采用的**sigmoid函数，线性函数结合sigmoid就成了LR模型**

$$
y = \sigma(f(x)) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}
$$
注意，**这里的w^T，x 都是已经包含了偏置项的，只需要在原来的基础上变成x_new = [x, 1] ,  w_new = [w，b]**

一个样本，它类别为1 的概率为：

$$
P_{y=1} = \frac{1}{1+e^{-w^Tx}}=p
$$
那么类别为0的概率

$$
P_{y=0} = 1-p
$$
一个事件发生也不发生的比例：

$$
\frac{P(y=1|x)}{P(y=0|x)} = \frac{p}{1-p} = e^{g(x)}
$$
称为事件发生比；记为odds

写作：

$$
P(y|x) = 
\begin{cases}
	p, \text{y=1}\\
    1-p, \text{y=0}
\end{cases}
$$
统一一下形式，转为一个公式表示就是：
$$
p(y_i|x_i) = p^{y_i}(1-p)^{1-y_i} \rightarrow 式1
$$
假设一组样本集有N个样本，那么这个样本集发生的概率就是单个样本发生概率的相乘，称为**联合分布，也是式1的\*似然函数\***：

$$
P总 = P(y1|x1)P(y2|x2)P(y3|x3)...P(yN|xN) = \prod_{i=1}^{N}p^{y_n}(1-p)^{1-y_n} \rightarrow 式2
$$
连乘形式不方便计算，两边同时取对数：

$$
\begin{equation*}
\begin{split}
F(w) = & ln(P总)\\
&= ln(\prod_{n=1}^{N}p^{y_n}(1-p)^{1-y_n})\\
&=\sum_{n=1}^{N}ln(p^{y_n}(1-p)^{1-y_n})\\
&=\sum_{n=1}^{N}(y_nln(p)+(1-y_n)ln(1-p)) \text 式3
\end{split}
\end{equation*}
$$
此时的**F(x)是LR模型的损失函数**，当然有的博客中的F(x)也可以除以N，不影响结果，但是为什么F(x)就是损失函数呢？

**这是由极大似然法推导而来**

**极大似然法**

如果用极大似然法求解式3，上式中只有p中带有一个w是未知量。**我们现在要做的就是找到一组合适的W使得样本集的联合分布概率最大**。解释来说就是，样本集这个事件已经真实发生了，我们倒过去，我们要找一组权重，尽可能的使得它发生的概率最大，这样，这一组w就尽可能的接近真实w。例如，已知一个学校男生比女生7:3 ，一个是3:7，现在抽一个人是男生，那么最可能来自哪个学校？当然是第一个学校，虽然可能出错，不过已经是已知条件下概率最大的结果了。

首先对权重n+1（因为包含了偏置b）分别求偏导，假设对wk求偏导：

![20140528205737500](/assets/img/ML/one-stop-machine-learning/max-likely.jpg)
![20140528210021312](/assets/img/ML/one-stop-machine-learning/max-likely2.jpg)

产生n+1 个等式，然后利用**牛顿法**迭代求解；

到现在为止我们还是没有解释为什么式3就是损失函数，直到我们不用极大似然估计和牛顿法去求解而是用梯度上升/下降法去求解，就很明显了。

**梯度上升法/下降法**

我们同时也可以用**梯度上升法**求解式3，求得一组w使得式3的达到最大值。或者我们在式3前面乘上一个负因子

![gradient-asc](/assets/img/ML/one-stop-machine-learning/gradient-asc.jpg)

那么梯度上升法，就转换为了梯度下降法。**此时问题就变成了寻找一组w使得因子加权后的似然函数最小，是不是很熟悉，这个因子加权后的似然函数，不就是机器学习中常见的损失函数嘛**

![20131113203723187](/assets/img/ML/one-stop-machine-learning/LR-loss-solu.jpg)

其中 g(x) = θT * x, 并且

![20131113203741453](/assets/img/ML/one-stop-machine-learning/LR-loss-solu2.jpg)

权重θ 更新函数：由于1/m是个常数，和学习率a合并

![20131113205240203](/assets/img/ML/one-stop-machine-learning/LR-loss-func3.jpg)

**梯度下降过程向量化**

见公式15，每次梯度更新都需要遍历m个样本，但是在实际应用中，往往不用for loop实现，而是用vector 内积实现for循环的效果

记θ*x 即样本输入点乘权重的结果，经过sigmoid函数之前

![20131113204012546](/assets/img/ML/one-stop-machine-learning/LR-gd.jpg)

基于向量A计算误差向量E：

![20131113204103593](/assets/img/ML/one-stop-machine-learning/LR-gd2.jpg)

有了误差向量E之后， 权重更新就可以实现向量化，省去for loop， θj可以表示为式23，整个权重θ统一表示为式24

![20131113204138093](/assets/img/ML/one-stop-machine-learning/LR-gd3.jpg)
![20131113204152062](/assets/img/ML/one-stop-machine-learning/LR-gd4.jpg)

**逻辑回归为什么要用sigmoid函数？能不能用其他函数？**

**为什么是sigmoid？**

首先想到是用W*X来表示属于类别1的概率，因为w*x的值越大离分类面越远，有两个问题

1. 这个值是负无穷到正无穷，需要归一化为0-1
2. 现实中，w*x 非常大或者非常小的时候对概率值影响不大，但是w*x 在0范围附近，也就是在分类面的附近对概率影响愈来愈大。离分类面很远，再远一点也对概率值理论上影响不大

于是需要修正，首先想到的思路是时间的几率odds = p/(1-p)是0到正无穷，同时为了满足第二个条件，对odds取log 就同时满足了2个条件

$$
log(\frac{p}{1-p}) = w^Tx \Longrightarrow p=\frac{1}{1+e^{-w^Tx}}
$$
**为什么只能是sigmoid？**

1.正态分布解释

我们不知道分类事件的发生符合什么分布形式，一般默认符合正态分布，正态分布的概率密度函数是钟形，但是其分布函数却是类似“sigmoid”的形状，而且sigmoid的求导方便又很近似正态分布，所以采用sigmoid

2.最大熵解释

该解释是说，在我们给定了某些假设之后，我们希望在给定假设前提下，分布尽可能的均匀。对于Logistic Regression，我们假设了对于{X,Y}，我们预测的目标是Y|X，并假设认为Y|X服从bernoulli distribution，所以我们只需要知道P(Y|X)；其次我们需要一个线性模型，所以P(Y|X)=f(wx)。接下来我们就只需要知道f是什么就行了。而我们可以通过最大熵原则推出的这个f，就是sigmoid

无论是sigmoid函数还是probit函数都是**广义线性模型的连接函数**（link function）中的一种。选用联接函数是因为，从统计学角度而言，**普通线性回归模型是基于响应变量和误差项均服从正态分布的假设，且误差项具有零均值，同方差的特性**。但是，例如分类任务（判断肿瘤是否为良性、判断邮件是否为垃圾邮件），其响应变量一般不服从于正态分布，其服从于二项分布，所以选用普通线性回归模型来拟合是不准确的，因为不符合假设，所以，我们需要选用广义线性模型来拟合数据，通过标准联接函数(canonical link or standard link function)来映射响应变量，如：正态分布对应于恒等式，泊松分布对应于自然对数函数，二项分布对应于logit函数（二项分布是特殊的泊松分布）。

**Reference**

[Logistic回归原理及公式推导](https://blog.csdn.net/AriesSurfer/article/details/41310525)

[机器学习--Logistic回归计算过程的推导](https://blog.csdn.net/ligang_csdn/article/details/53838743)

[逻辑斯蒂回归原理篇](https://blog.csdn.net/a819825294/article/details/51172466 )

[sigmoid函数与softmax函数](https://www.jianshu.com/p/52fcd56f2406)

### KNN

KNN是一种**基于instance的学习算法**，基于instance的学习方法只是简单的把训练样本存储起来。每当学习器遇到一个新的instance，分析新的实例和以前存储实例的关系，据此把一个目标函数值赋给新实例

基于实例方法的不足：

- 分类时的开销大，所有的计算基本都发生在分类的时候
- 当从存储器中检索相似的训练样例时，它们一般考虑实例的所有属性。如果目标概念仅依赖于很多属性中的几个时，那么真正最“相似”的实例之间很可能相距甚远。

KNN假设实例是n维空间中的一个点，，点与点的距离是由标准欧式距离定义的,当然也可以用Lp距离来定义

在新样本输入，需要进行分类时候，对已知的每个样本点计算距离，取前k个样本中个数最多的类别作为新样本的类别，也可以进行加权求和最为新样本的类别或者回归问题的输出。

改进型有weighted knn，根据距离进行贡献加权，很好理解。

**KNN算法实现，KD树**

![WeChat Screenshot_20190811181452](/assets/img/ML/one-stop-machine-learning/knn-kd.png)

![WeChat Screenshot_20190811181503](/assets/img/ML/one-stop-machine-learning/knn-kd2.png)

有几个重要的点，划分维度是循环的，假设有2个特征维度，6个数据点，kd树大概3层，第一层按照x轴划分，左右子集分别按照y轴划分，他们的左右子集又按照x轴划分，中位数所在的那个样本留在根节点，不被分为左右样本，这样每个节点至少有一个样本、

搜索kd树的算法

![081118282872_0WeChat Screenshot_20190811182745](/assets/img/ML/one-stop-machine-learning/kd-search.png)

在建树的过程中， 每个叶子节点相当于分配了一个区域

首先找到所在叶子节点，按照输入节点和所在叶子节点画球，往上遍历到父节点，如果父节点的另一个子节点的区域和这个球相交，说明可能这个区域内有更近的，遍历整个区域的点，如果有，更新最近点，然后再按照输入节点和最近的点画球。直到访问到最上面的根节点

**特点**：

- 对样本量大的类别有偏向，但是在样本量足够大的情况下，效果不错。由于是k个加权平均，robustness 强
- 距离容易被不相关的属性支配。距离是基于所有属性计算的，有些feature对于类别的重要性比较大，可能就在计算中导致偏差较大，可以考虑对属性进行加权
- k值的选择对结果的影响很大，如果k值越小，相当于用越少的邻域中的训练实例进行预测，只有很相似的样本点参与预测，近似误差小，当前于在测试集上表现良好，模型越复杂，但是估计误差会增大，容易受噪声点支配；k越大，相当于在越大的邻域中进行预测，一些不相关的点也进来预测，近似误差大，但是估计误差好一点，模型越简单，极限情况下，k=N，模型一直返回多数类，相当于过于简单，返回恒定值

**Reference**

[数据挖掘十大算法--K近邻算法](https://wizardforcel.gitbooks.io/dm-algo-top10/content/knn.html)

### 朴素贝叶斯

朴素贝叶斯属于生成式模型，学习输入和输出的联合概率分布。给定输入x，利用贝叶斯概率定理求出最大的后验概率作为输出y。

首先肯定是贝叶斯公式：

![img](/assets/img/ML/one-stop-machine-learning/by.png)

P(B|A)称为后验概率 P（A|B）称为似然函数，P(A),P(B)为先验概率

朴素贝叶斯的思想很简单：给出一个待分类项，出现这个待分类项的情况下，各个类别出现的概率，那个高就取那个类别作为分类结果；就和看到一个黑人，预测他的国家，假设我们现在数据集中，70%黑人是非洲，20%美洲，10%欧洲，那肯定预测非洲更加准确。

朴素贝叶斯之所以叫naive bayes，是因为它假设了x的各个属性之间条件独立，这个假设很胆大，在很多情况下是不成立的，所有朴素贝叶斯的效果很多时间都不太好。

换言之，该假定说明给定实例的目标值情况下，观察到联合的*a*1,*a*2…*an*的概率正好是对每个单独属性的概率乘积：

![img](/assets/img/ML/one-stop-machine-learning/nb1.jpg)

**朴素贝叶斯分类的正式定义如下：**

1、设![img](/assets/img/ML/one-stop-machine-learning/by2.gif)为一个待分类项，而每个a为x的一个特征属性

2、有类别集合![img](/assets/img/ML/one-stop-machine-learning/by3.gif)

3、计算![img](/assets/img/ML/one-stop-machine-learning/by4.gif)

4、如果![img](/assets/img/ML/one-stop-machine-learning/by5.gif)，则![img](/assets/img/ML/one-stop-machine-learning/by6.gif)

那么现在的关键就是如何计算第3步中的各个条件概率。我们可以这么做：

1、找到一个已知分类的待分类项集合，这个集合叫做训练样本集。

2、统计得到在各类别下各个特征属性的条件概率估计。即![img](/assets/img/ML/one-stop-machine-learning/by7.gif)。

3、如果各个特征属性是条件独立的，则根据贝叶斯定理有如下推导：

![img](/assets/img/ML/one-stop-machine-learning/by8.gif)

因为分母对于所有类别为常数，因为我们只要将分子最大化皆可。又因为各特征属性是条件独立的，所以有：

![img](/assets/img/ML/one-stop-machine-learning/by9.gif)

**Reference**

[朴素贝叶斯分类器](https://wizardforcel.gitbooks.io/dm-algo-top10/content/naive-bayes.html)



## 聚类

聚类不同于分类算法有一个最优化目标，而是一种统计方法，把相似的数据聚在一起。 在进行聚类之前，有必要对数据有一个初步的理解，如果数据是纯随机分布的话，虽然任何一个聚类算法都可以“强行”得到一个结果，但是这个聚类效果是没有意义的。

Assessing clustering tendency :

Hopkins statistic 霍普金斯统计量，越接近1，聚类约有意义

[python hopkins statistic 实现](https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/)

### **K-means**

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
2. 如果一个点在两个核心对象的邻域内，但是自己不是核心对象，遵守先来后到的原则划分聚类簇

**适用性和优缺点：**

- 适用于样本密度大，且非凸的情况
- 在聚类的同时发现outlier，对异常值不敏感
- 初始化不影响总体结果，但是也不是完全稳定的，存在先来后到的情况，不同的选取顺序也会有影响
- 密度不均匀，类内间距大的时候不适用
- 于k-means相比对了一个参数需要调参，邻域范围和最小样本阈值都需要调参，且影响很大



### **GMM(Gaussian mixture model)**

 to be completed



## 模型之间的异同

### 分类/回归

#### LR和SVM的异同

- LR是参数模型，SVM是非参数模型
- Linear SVM不直接依赖数据分布，分类平面不受一类点影响；LR则受所有数据点的影响，如果数据不同类别处于极其不平衡的状态, 一般需要先对数据做平衡处理。
- Linear SVM依赖数据表达的距离测度，所以需要对数据先做标准化；LR不受其影响
- Linear SVM依赖惩罚项的系数，实验中需要做[交叉验证](https://zhuanlan.zhihu.com/p/32627500)
- Linear SVM和LR的执行都会受到异常值的影响，其敏感程度而言，谁更好很难下明确结论。
- Linear SVM和LR损失函数不同, LR为logloss, SVM为hinge loss. 而SVM中的
  称为**hinge loss**。
- LR 能做的 SVM都能做，但可能在准确率上有问题，SVM能做的LR有的做不了。

#### LR和线性回归的区别

线性回归用来做预测,LR用来做分类。线性回归是来拟合函数,LR是来预测函数。线性回归用最小二乘法来计算参数,LR用最大似然估计来计算参数。线性回归更容易受到异常值的影响,而LR对异常值有较好的稳定性。

#### GBDT和随机森林的异同：

相同

- 都是由多棵树组成
- 最终的结果都是由多棵树一起决定

不同

- 组成随机森林的树可以是分类树，也可以是回归树；而GBDT只由回归树组成；

- 组成随机森林的树可以并行生成；而GBDT只能是串行生成；

- 对于最终的输出结果而言，随机森林采用多数投票等；而GBDT则是将所有结果累加起来，或者加权累加起来；

- 随机森林对异常值不敏感，GBDT对异常值非常敏感；

- 随机森林对训练集一视同仁，GBDT是基于权值的弱分类器的集成；

- 随机森林是通过减少模型方差提高性能，GBDT是通过减少模型偏差提高性能。

#### GBDT 和XGBoost的不同

- GBDT的梯度拟合只考虑了一阶，XGBoost做了泰勒2阶展开

- XGBoost损失函数加入正则项，正则函数系数 *节点个数+系数*（叶子节点值的L2范数和）

- XGBoost支持并行化

#### XGBoost, LR, RF的优缺，适用场景

XGBoost：

优缺点：1）在寻找最佳分割点时，考虑传统的枚举每个特征的所有可能分割点的贪心法效率太低，XGBoost实现了一种近似的算法。大致的思想是根据百分位法列举几个可能成为分割点的候选者，然后从候选者中根据上面求分割点的公式计算找出最佳的分割点。2）XGBoost考虑了训练数据为稀疏值的情况，可以为缺失值或者指定的值指定分支的默认方向，这能大大提升算法的效率，paper提到50倍。3）特征列排序后以块的形式存储在内存中，在迭代中可以重复使用；虽然boosting算法迭代必须串行，但是在处理每个特征列时可以做到并行。4）按照特征列方式存储能优化寻找最佳的分割点，但是当以行计算梯度数据时会导致内存的不连续访问，严重时会导致cache miss，降低算法效率。paper中提到，可先将数据收集到线程内部的buffer，然后再计算，提高算法的效率。5）XGBoost还考虑了当数据量比较大，内存不够时怎么有效的使用磁盘，主要是结合多线程、数据压缩、分片的方法，尽可能的提高算法的效率。

适用场景：分类回归问题都可以。

RF：

优点：1）表现性能好，与其他算法相比有着很大优势。2）随机森林能处理很高维度的数据（也就是很多特征的数据），并且不用做特征选择。3）在训练完之后，随机森林能给出哪些特征比较重要。4）训练速度快，容易做成并行化方法(训练时，树与树之间是相互独立的)。5）在训练过程中，能够检测到feature之间的影响。6）对于不平衡数据集来说，随机森林可以平衡误差。当存在分类不平衡的情况时，随机森林能提供平衡数据集误差的有效方法。7）如果有很大一部分的特征遗失，用RF算法仍然可以维持准确度。8）随机森林算法有很强的抗干扰能力（具体体现在6,7点）。所以当数据存在大量的数据缺失，用RF也是不错的。9）随机森林抗过拟合能力比较强（虽然理论上说随机森林不会产生过拟合现象，但是在现实中噪声是不能忽略的，增加树虽然能够减小过拟合，但没有办法完全消除过拟合，无论怎么增加树都不行，再说树的数目也不可能无限增加的）。10）随机森林能够解决分类与回归两种类型的问题，并在这两方面都有相当好的估计表现。（虽然RF能做回归问题，但通常都用RF来解决分类问题）。11）在创建随机森林时候，对generlization error(泛化误差)使用的是无偏估计模型，泛化能力强。

缺点：1）随机森林在解决回归问题时，并没有像它在分类中表现的那么好，这是因为它并不能给出一个连续的输出。当进行回归时，随机森林不能够做出超越训练集数据范围的预测，这可能导致在某些特定噪声的数据进行建模时出现过度拟合。（PS:随机森林已经被证明在某些噪音较大的分类或者回归问题上回过拟合）。2）对于许多统计建模者来说，随机森林给人的感觉就像一个黑盒子，你无法控制模型内部的运行。只能在不同的参数和随机种子之间进行尝试。3）可能有很多相似的决策树，掩盖了真实的结果。4）对于小数据或者低维数据（特征较少的数据），可能不能产生很好的分类。（处理高维数据，处理特征遗失数据，处理不平衡数据是随机森林的长处）。5）执行数据虽然比boosting等快（随机森林属于bagging），但比单只决策树慢多了。

适用场景：数据维度相对低（几十维），同时对准确性有较高要求时。因为不需要很多参数调整就可以达到不错的效果，基本上不知道用什么方法的时候都可以先试一下随机森林。

LR：

优点：实现简单，广泛的应用于工业问题上；分类时计算量非常小，速度很快，存储资源低；便利的观测样本概率分数；对逻辑回归而言，多重共线性并不是问题，它可以结合L2正则化来解决该问题。

缺点：当特征空间很大时，逻辑回归的性能不是很好；容易欠拟合，一般准确度不太高

不能很好地处理大量多类特征或变量；只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；对于非线性特征，需要进行转换。

适用场景：LR同样是很多分类算法的基础组件，它的好处是输出值自然地落在0到1之间，并且有概率意义。因为它本质上是一个线性的分类器，所以处理不好特征之间相关的情况。虽然效果一般，却胜在模型清晰，背后的概率学经得住推敲。它拟合出来的参数就代表了每一个特征(feature)对结果的影响。也是一个理解数据的好工具。

### 聚类

#### K-Means 和 DBSCAN的区别

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

### 其他

#### 生成式模型和判别式模型的区别

简单来说，判别式模型就是一个模型，输入进去，label就输出；生成式模型是很多个模型组合，一般类别有多少就有多少模型，例如朴素贝叶斯，需要算每一个类的概率值，然后取最大概率值的类作为label

用统计学的角度来看，生成式模型学习了输入和输出的联合概率分布P（x，y）对于输入x来说，每个y的概率是多大，都可以知道；判别式模型学习的是条件概率分布P（y|x）

优缺点：

1. 生成式模型通常对数据分布会做一定的假设，在假设成立时，生成式模型可以用较少的数据取得不错的效果，但是不成立时，判别式模型的效果更好
2. 由于现实中数据分布的假设一般是不成立的，所以判别式模型的错误率会比生成式模型低，但是生成式模型需要更少的数据量来让错误率收敛
3. 生成式模型更容易拟合，判别式模型需要解决凸优化的问题
4. 新增类别的时候，判别式模型需要重新训练，生成式模型不需要

#### 参数模型和非参数模型

在统计学中，参数模型通常假设总体（随机变量）服从某一个分布，该分布由一些参数确定（比如正太分布由均值和方差确定），在此基础上构建的模型称为参数模型；非参数模型对于总体的分布不做任何假设，只是知道总体是一个随机变量，其分布是存在的（分布中也可能存在参数），但是无法知道其分布的形式，更不知道分布的相关参数，只有在给定一些样本的条件下，能够依据非参数统计的方法进行推断。