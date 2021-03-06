---
layout: post
title: 推荐系统
category: DeepLearning
tags: recommendation system
description: 推荐系统模型，算法，框架，相关论文
---

<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
</head>

## 基础知识

### **目的**

- 激发用户兴趣，促使用户去做某件事情，例如购物，看电影。<br>
- 解决信息过载问题，从大集合中寻找出合适用户的内容。<br>

所以推荐系统同时涉及到**信息检索**和**信息过滤**，不同于搜索场景，有确定的query，用户需求明确； 推荐系统是没有用户的query，需要根据用户画像，内容画像从总体内容池中选出多样性的偏好内容。

## Framework

### Recall Layer

常用模型： DSSM, YouTube DNN

**DSSM模型**（Deep Structure Semantic Model）是有微软提出的用于网页搜索的深度网络模型。

![img](/assets/img/recommendation_system/dssm_arch.png)

原生的DSSM 结构如上，两个深度网络结构分开，深度网络的拓扑结构可以更换，DNN, LSTM 等都可以，其中一个深度网络负责把user的query vector映射到一个低维空间，一个负责把doc vector 映射到和query vector同一个低维空间下。对于每一个<query vector', doc vector'>计算距离，一般采用cosine距离，对最后计算的距离排序就是一定程度上query和doc的匹配程度排序。对于同一类doc 可以共用一个item 侧深度网络，如果需要同时输入不同类型的doc，例如图文和视频，则需要相同结构或是不同结构的不同item侧深度网络，如果此时共用一个user侧深度结构，那么就是dssm的改进型 Multi-view DSSM。

![img](/assets/img/recommendation_system/mv_dssm_arch.jpg)

**YouTube DNN**

![img](/assets/img/recommendation_system/youtube-dnn.png)

**模型组成**<br>
两个DNN nerual network结构； candidate generation 的输入是用户历史行为，输出是召回的small subset of video from a large corpus, 通过collaborative filtering 进行简单的相关召回

**模型细节**<br>
把推荐任务看成极限多分类任务；即在时刻*t*，为用户*U*（上下文信息*C*）在视频库*V*中精准的预测出视频*i*的类别（每个具体的视频视为一个类别，*i*即为一个类别). <br>
candidate generation 部分的模型结构图

![ytb-dnn](/assets/img/recommendation_system/youtube-dnn-arch.png)

三层DNN 结构，激活函数式ReLU；训练时，初始化一个video vectors 组成的矩阵。假设最后一层DNN 的输出是1\*64维，video matrix的维度是64\*80w维，则多分类的输出是一个1\*80w维的vector，代表着某个sample被划为80w个不同类别的概率，概率最大的输出是最终结果，和用户实际点击的video计算loss，反向传播，更新权重。线上服务的时候，某个sample经过DNN 结构生成user-vector（整个DNN 结构可以当成一个raw input到user-vector的映射），线上服务的时候，video vector是已经train好了的，直接访问就行。user vector的1\*128维 和128\*80w做knn操作，取出80w个列向量中和user-vector最相似的top k个。

**训练集细节：**用户看过的video set是变长的，通过embedding 把每个video map到一个定长的vector，这些vector之间对应位置的值进行“averaging”形成最终代表user watched videos的一个定长的vector，user search 进行同样的操作。其他特征中比较重要的是人口统计学信息，使模型一定程度上可以用于用户冷启动；二元和continuous feature map到[0-1], 在训练过程中，**embedding matrix 是和DNN 网络参数一起梯度下降的**。对于每个IDs Space 都有一个单独的embedding，大小和IDs space的大小成正比。

Ranking部分模型结构图

![ytb-dnn-rank](/assets/img/recommendation_system/youtube-dnn-ranking.png)

**与召回层不同的是training 的最后一层是LR结构,** **训练目标大体是期望观看时长**，如果以CTR为优化目标，会偏向于推荐具有迷惑性的video，那么用户虽然点击但是没有观看完的video. 线上服务的时候使用e^(wx+b)作为激活函数，是期望停留时长的近似，也就是某个video最后的score. **Label 是曝光的视频是否点击，如果点击，则annotated with 浏览时长**


 **衡量召回策略好坏的两个指标：**<br>
-  **召回率**：正确召回的/应该被召回的
-  **准确率**：正确召回的/召回总数

**召回策略主要有三种**

- **基于内容的召回**：分为基于内容标签的召回：将内容画像和用户画像相匹配，基于用户历史浏览过的内容标签，推荐相   似标签的内容； 基于知识的匹配： 基于先验知识，得知某一内容和用户之前浏览过的内容有相关性，  进行召回。

- **基于协同过滤的召回（user-based, item-based, model-based）**
  user-based: 发现相似用户，进行用户内容的交叉召回， 这种方式的难点在于用户相似度的衡量和近邻数量的选择上，同时用户量很大的情况下，如何实时的计算相似度也是一个难题。
  item-based：计算item之前的相关性，召回相关度最高的item，这种方式对比user-based的好处在于item之间的相似度可以线下预处理，在特征维度特别大的情况下，实时性好。
  model-based: 训练模型， 根据用户的实时爱好进行召回

- **基于知识的召回**
  例如已知复联2是复联的续集，那么看过复联的用户对复联2的兴趣从常理的角度看应该是很高的，基于这个考虑，加上已经提前知道的“知识”，进行基于知识的召回。


### Score Layer

打分层常用模型：wide&Deep , DNN，

**Wide&Deep 模型**

![Deep&Wide模型结构图](/assets/img/recommendation_system/wide&deep.png)

**为什么要组合wide && deep 模型？**

**Pros：**<br>
- Memorization(记忆)：学习出现的规律，从发生过的记录中学习规律，可以通过a wide set of crossed feature 交叉特征来学习，LR模型加上大量的特征工程可以很好学习已知的特征组合。
- Generalization(泛化)：学习unseen feature combination，通过embedding操作实现，embedding操作一般在DNN中使用
- 组合wide and deep 结构的模型，使得模型同时具有拟合过去历史的能力和一定的泛化能力。

**Cons：**<br>
- embedding 操作可能会造成over-generalized 的情况。当user-item interaction are sparse and high-rank的时候，embedding操作会使得很多原本相关性不大的事物被推送给用户。
- DNN 的解释性一直是一个问题，假设原始输入特征100w维，1000w个sample，假设embedding matrix的维度为100w\*128，DNN网络的输入是1000w*128维的matrix，embedding操作将100w维映射到了128维，这128维代表的含义是不知道的。整个embedding层加上后续的DNN 网络可以看成一个黑匣子，100w维特征的1000w个sample输入，输出了1000w个结果。单个特征对于结果的影响是不知道的，DNN 网络中权重不代表特征对于结果的贡献度。LR则是interpretable的，最后的结果就是特征值加权求和的结果，特征对于最后结果的贡献程度取决于权重。
- LR虽然解释性强，但是需要大量的人工去进行特征组合，LR “记忆”能力强，泛化能力弱，可以通过增加一些泛化的组合特征增强LR的泛化能力（例如：用户看过军事&&用户看过历史）

**论文笔记**

|          | wide侧                                                       | deep侧                                                       |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 模型细节 | generalized linear model（LR）特征有raw input feature 和组合特征 | DNN structure with embedding matrix deep 和 wide 侧通过weighted sum + bias进行连接，取sigmoid，as prediction |
| 优化方法 | Follow-the-regularized-leader(FTRL) algorithm with L1 regularization | AdaGrad                                                      |
| 特征     | 用户安装的app和曝光过的app记录                               | **原始特征**包含user安装过的app曝光过的app，人口统计学信息，基本信息，device class等，对于每个categorical feature 有一个32维的embedding |

**训练方式**<br>
Joint training 而不是Ensemble；Joint training 和 ensemble的区别在于joint training同时训练joint model，把joint model 当成一个模型来梯度下降，wide侧只需要补充deep部分memorization能力的不足就行，不需要一个庞大的feature set。ensemle 训练方式是两个模型单独训练，同时需要达到各自的期望效果，那么单独一个模型通常需要增大feature数量或者模型深度才可以达到一个reasonable的accuracy。

**训练数据**<br>
**label：** 用户是否安装了某个app（google play）**vocabularies：** 特征ID 化，把categorical类型特征映射到integer ID（对于那些满足最低出现次数的特征），continuous 特征映射到[0-1]

**Reference**<br>[模型paper地址](https://arxiv.org/pdf/1606.07792.pdf)

### Ranking layer

经过召回队列返回的文章还是有千甚至万，十万的量级，可是用户一次刷新只能返回十篇文章，怎么从万篇文章中找出这十篇文章就是打分层需要干的事情了。涉及到的领域就是Learning to rank，排序学习。

什么样的排序是一个好排序，需要指标去衡量，一篇文章排在第一位而不是第二位，需要背后的数字去支撑。在搜索环境下，用户有具体的query，衡量一个返回结果的好坏，看结果和query的相似度，越相似的结果理应排在最前面；与搜索场景类似但也有不同，信息流推荐场景下，一个文章的位置要综合多方面的考虑，是把点击率最高的文章放在前面，还是期望停留时长最长的放在前面，亦或是多方面的考虑，出于商业化的需要，返回结果中往往掺杂着广告，商业化和用户满意度的权衡也是一个很重要的考虑点。

推荐结果指标化主要有两个方面：
1： 衡量单个推荐结果本身的好坏（例如推荐结果多少程度上符合搜索query）
2：衡量返回10篇文章整体的好坏（10篇文章固定的情况下，如何排列才是最好的结果）

**指标的演化流程从P-R， Discounted Cumulative Gain, Expected Reciprocal Rank** 。

- **P-R** 准确率的定义和recall层相同，准确率是相对于召回结果（被认为是相关的文章，可能是真相关，也可能是假相关）来说，召回的结果中正确的比率，所以叫准确率， 召回率是对整个内容库（内容库是所有和query相关的文章）来说的，对于一个request，召回结果中正确的数量占总内容库的比例。**但是P-R 没有考虑到位置因素，TRUE,FALSE的分类也太过粗糙。**
- **Discounted Cumulative Gain** 假设一屏有p篇文章，第一篇文章和query的相似度为rel1，第二篇为rel2，总列表的指标就是下面公式加权出来的结果，位置越靠前的文章的相似度对总分的影响最大。
  ![img](/assets/img/recommendation_system/discounted_cumulative_gain.png)
- **Expected Reciprocal Rank** 在DCG的基础上考虑了前面文章的相关性
  ![img](/assets/img/recommendation_system/ERR.png)
  (1-Ri)代表文章的相关度，如果前面的文章的不相关度越大，那么该文章的相关度对总分的影响就越大。

排序层的样本生成方式可以分为： **pointwise, pairwise, Listwise**。pointwise 给每个样本一个具体的分数，可以是点击率或者其他数值。pairwise： 没有具体的数值，但是知道任何两个样本之间的好坏关系。Listwise： 一个列表，包含了所有样本的好坏排序，但是具体好多少，或者坏多少，不知道。

**在线排序分层模型**
![img](/assets/img/recommendation_system/online-rank.png)

scene dispatch： 场景分发，划分不同的业务类型<br>
traffic distribution： 流量分发，包含模型分发和流量分桶，将总流量分为不同的桶，对每个模型也分为一定配额的桶，如果流量和模型分到同一个桶，那么该部分的流量就会走到该模型。

## Algorithm

### 关联规则挖掘

关联规则描述两个不同事物之间的关联性，假设有两个非空集合x,y 存在 X->Y 则称之为一条关联规则,关联规则的强度由**支持度（support）**和**自信度（confidence）**描述.<br>
- **相对支持度support** (x --> y) （x并y）/总样本 <br>
- **相对自信度confidence**（x --> y）（x并y）/包含x样本数<br> 
- **绝对支持度**就是集合在样本中出现过的次数 = 相对支持度*总样本数<br>

**关联规则挖掘就是找出 support> support threshold的规则 confidence> confidence threshold的规则**， 如果使用穷举法，穷举所有可能的规则，计算量会爆炸，假设一个样本数为N的集合，所有可能的组合Cn1+Cn2 +...+ Cnn-1 = 2^n - 1

关联规则的挖掘大体分两步，给定最小支持度和自信度的前提下<br>
1. 生成频繁项集（支持度大于最小支持度的集合）<br>
2. 在频繁项集的基础上筛选出满足最小自信度的项集
   由于频繁项集的数据量不会很大，所以第二步的运算时间相对较短，主要time complexity 在第一步。

常用Aprior算法简化运算量，aprior算法的两个主要简化思想<br>
1：如果一个集合是频繁项集，那么它的所有子集合都是频繁项集，例如假设{A,B,C,D}是频繁项集，满足支持度大于最小支持度阈值，那么任何子集例如{A,B,C}的覆盖度一定是比{A,B,C,D}要大的，所以一定满足最小支持度，一定是频繁项集<br>
2：如果一个集合不是频繁项集，即不满足最小支持度，那么任何超集都不是频繁项集，例如假设{A,B}不是频繁项集，那么任何包含{A,B}的集合都不会是频繁项集，超集的覆盖度一定是比当前集合要小的，所以一定不会满足最小支持度<br>

aprior算法的流程大概如下，是一个迭代的过程，一直到不存在新的超集满足要求，假设支持度最低要求为大于等于3，首先计算单个集合的支持度，剔除掉不满足的单个集合，之后进行自由组合，统计新的二元集合的支持度，剔除掉不满足要求的，图中剔除掉了{牛奶，啤酒}，{面包，啤酒}, 利用aprior算法的第二个简化规则，在生成三元集合的时候，任何同时包含{牛奶，啤酒}和{面包，啤酒}的三元集合都剔除掉，不需要计算支持度，因为这些集合是非频繁项集的超集，一定不是频繁项集。最后只剩下一个3元项集，算法迭代结束。在算法迭代过程中，剔除操作之后剩下的集合都是频繁项集，这些集合的数量和所有可能的组合的个数相比，会小很多。在第二阶段的时候，在对这些频繁项集计算他们的自信度，最后筛选出目标集合。

![img](/assets/img/recommendation_system/relation_mining.jpg)

**Reference**<br>[数据挖掘系列（1）关联规则挖掘基本概念与Aprior算法](https://www.cnblogs.com/fengfenggirl/p/associate_apriori.html)<br>[Fast algorithm for mining association rules](http://rakesh.agrawal-family.com/papers/vldb94apriori.pdf)

### Content Based

**基于Item本身属性**的推荐，计算物品之间的相关性，然后根据用户的历史爱好，推荐给用户相似的物品。主要涉及到三个步骤。1，构造物品特征；2，计算物品之间的相似度；3，判断用户爱好。

**优缺点**: 推荐的质量取决于对物品建模的完整和全面程度，仅仅考虑了物品各属性之间的相关性，效果有限，且不适用于冷启动。但是该方法与用户行为独立，可以推荐新产生，用户还没有行为的物品，避免热门物品被反复推荐。

**Reference**<br>[推荐机制 协同过滤和基于内容推荐的区别](https://www.cnblogs.com/fengff/p/10187150.html)

### Collaborative Filtering

**协同过滤的方法是基于用户的**,不考虑物品本身的属性，协同过滤又可以分为三个子类，User Based, Item Based, Model Based.

#### UserCF

基于用户对商品的行为，计算行为相似的用户，推荐相似用户的商品给他。首先计算用户的**评分矩阵** m\*n m是用户数，n是物品数。$d_{ij}$表示用户$i$对$j$商品的行为的总分，收藏，购买，点赞等行为都有对应的分数；例如：

| 用户/商品 | 1    | 2    | 3    | 4    | 5    | 6    |
| :-------- | :--- | :--- | :--- | :--- | :--- | :--- |
| A         | 1    |      | 5    | 3    |      |      |
| B         |      | 3    |      |      | 3    |      |
| C         | 5    |      |      |      |      | 10   |
| D         | 10   |      |      |      | 5    |      |
| E         |      |      | 5    | 1    |      |      |
| F         |      | 5    | 3    |      |      | 1    |

第二部计算用户相似度，一般采用cosine距离来衡量两个用户的相似度,计算用户**相似度矩阵**m\*m

|      | A    | B    | C    | D    | E    | F    |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A    | 1    | 0    | 0.08 | 0.15 | 0.93 | 0.43 |
| B    | 0    | 1    | 0    | 0.32 | 0    | 0.6  |
| C    | 0.08 | 0    | 1    | 0.4  | 0    | 0.15 |
| D    | 0.15 | 0.32 | 0.4  | 1    | 0    | 0    |
| E    | 0.93 | 0    | 0    | 0    | 1    | 0.5  |
| F    | 0.43 | 0.6  | 0.15 | 0    | 0.5  | 1    |

第三部，计算推荐列表，**推荐列表（m\*n）=相似度矩阵（m\*m）x 评分矩阵（m\*n）**

|      | 1    | 2    | 3    | 4    | 5    | 6    |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A    | 2.9  | 2.2  | 11.0 | 3.9  | 0.8  | 1.2  |
| B    | 3.2  | 6.0  | 1.8  | 0    | 4.6  | 0.6  |
| C    | 9.1  | 0.8  | 0.9  | 0.2  | 2.0  | 10.2 |
| D    | 11.2 | 1.0  | 0.8  | 0.5  | 6.0  | 4.0  |
| E    | 0.9  | 2.5  | 11.2 | 3.8  | 0    | 0.5  |
| F    | 1.2  | 6.8  | 7.7  | 1.8  | 1.8  | 2.5  |

第四步，由于用户之前已经对一些商品有过行为，所以把那些商品去掉，得到最后的推荐列表，取前k个最相关的商品推荐给用户

|      | 1    | 2    | 3    | 4    | 5    | 6    |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A    |      | 2.2  |      |      | 0.8  | 1.2  |
| B    | 3.2  |      | 1.8  | 0    |      | 0.6  |
| C    |      | 0.8  | 0.9  | 0.2  | 2.0  |      |
| D    |      | 1.0  | 0.8  | 0.5  |      | 4.0  |
| E    | 0.9  | 2.5  |      |      | 0    | 0.5  |
| F    | 1.2  |      |      | 1.8  | 1.8  |      |

#### ItemCF

基于用户对商品的偏好，把相似商品推荐给他，这里的**相似商品并不是由商品本身的属性决定的，而是用户的行为作为商品的“属性”去计算相似度**，说到底还是只考虑用户行为，而不考虑物品本身的属性。
计算过程也非常相似，区别在于计算时把UserCF的**评分矩阵转置**，再计算商品与商品之间的相似度得到**商品之间的相似度矩阵**。
最后的**推荐列表 = 商品之间的相似度矩阵 X 评分矩阵转置**

#### 对比

Item CF 和 User CF 是基于协同过滤推荐的两个最基本的算法，User CF 是很早以前就提出来了，Item CF 是从 Amazon 的论文和专利发表之后（2001 年左右）开始流行，大家都觉得 Item CF 从性能和复杂度上比 User CF 更优，其中的一个主要原因就是对于一个在线网站，用户的数量往往大大超过物品的数量，同时物品的数据相对稳定，因此计算物品的相似度不但计算量较小，同时也不必频繁更新。但我们往往忽略了这种情况只适应于提供商品的电子商务网站，对于新闻，博客或者微内容的推荐系统，情况往往是相反的，物品的数量是海量的，同时也是更新频繁的，所以单从复杂度的角度，这两个算法在不同的系统中各有优势，推荐引擎的设计者需要根据自己应用的特点选择更加合适的算法。

在非社交网络的网站中，内容内在的联系是很重要的推荐原则，它比基于相似用户的推荐原则更加有效。在社交网络站点中，User CF 是一个更好错的选择，User CF 加上社会网络信息，可以增加用户对推荐解释的信服程度。

#### 优缺点

与基于内容的推荐不同，CF不需要对物品严格建模，与物品所处领域无关，但是是基于历史数据的不适用冷启动问题，对于长尾审美，也不能很好的推荐。UserCF计算的是user之间的相似度，更加注重用户所在的兴趣小组，更加注重社会化;ItemCF注重的是用户有过行为的历史物品，所以更加个性化。

在单用户的多样性上，ItemCF比不上UserCF,因为它是基于用户之前有过行为的物品进行的推荐，覆盖面太小；但是在系统层面，ItemCF比UserCF的多样性要好，UserCF倾向于只推荐热门物品， ItemCF可以挖掘出长尾物品。

同时UserCF, ItemCF也各有限制，UserCF推荐的假设是，用户喜欢那些和他有相同爱好的用户喜欢的东西，推荐效果取决于用户有多少"相似用户"，如果一个用户有长尾爱好，则推荐效果可能欠佳。ItemCF推荐的假设是用户喜欢和他之前有过行为的商品相似的商品，如果用户行为的自相似度高，说明假设成立，如果反之，则假设不成立，推荐效果欠佳。

**Reference**<br>[推荐系统UserCF, ItemCF](https://www.jianshu.com/p/ec3de12db6e7)

### LFM

Latent Factor Model隐性因子模型，意在找出用户的兴趣因子。LFM对用户喜欢的物品类别，和物品分属于类别的概率进行建模，如下三个矩阵

![lfm](/assets/img/recommendation_system/lfm.jpg)

R：用户对物品的偏好信息，$R$ 表示用户$i$对物品$j$的偏好程度。P：表示用户对物品类别的偏好矩阵。Q：表示物品属于个类别的比例。从上述图中可以看出用户对物品的兴趣度可以用如下公式表示，推荐兴趣度top N的物品给用户即可。

$$
R(u,i) = \sum_{k=1}^{K}P_{u,k}Q_{i,k}
$$

那么问题来了，class类别数怎么分，P和Q矩阵如何确定。首先class类别数是一个hyper parameter，由先验确定，或者根据推荐结果，进行实验修改。P,Q矩阵参数首先是随机初始化，通过梯度下降方法进行优化迭代，模型学习出来的。那label又是哪里来的呢？R矩阵中如果用户对某个物品有过行为则为1，否则为0，如果推荐场景中能很好的收集**用户负反馈信息**(通常很难获取)，则行为可以为负数。损失函数顺理成章的是

$$
cost = \sum_{(u,i)\in S}(R_{ui}-\hat{R_{ui}})^2 +\lambda||P_u||^2 +\lambda||Q_i||^2
$$

对两个未知参数求梯度

$$
\frac{\partial c}{\partial P_{uk}} = -2\sum_{(u,i)\in S}(R_{ui}-\sum_{k=1}^{K}P_{u,k}Q_{k,i})Q_{ki}+2\lambda P_{uk} \\
\frac{\partial c}{\partial Q_{ki}} = -2\sum_{(u,i)\in S}(R_{ui}-\sum_{k=1}^{K}P_{u,k}Q_{k,i})P_{uk}+2\lambda Q_{ki}
$$

参数更新

$$
P_{uk} = P_{uk} +\alpha(\frac{\partial c}{\partial P_{uk}})\\
Q_{ki} = Q_{ki} +\frac{\partial c}{\partial Q_{ki}}
$$

从整个训练过程中可以看出，我们最终是想要矩阵R，也就是用户对于每个item的兴趣值，在这个过程中，我们得到中间结果矩阵R，也就是用户对于类别的喜好，以及矩阵Q，item属于类别的概率，这里相当于完成了item的自动“聚类”，我们不关心“聚类”的维度划分过程，只需要提供超参也就是类别个数。实际应用中，LFM通常是天级别更新。

**Reference**<br>[使用LFM（Latent factor model）隐语义模型进行Top-N推荐](https://blog.csdn.net/HarryHuang1990/article/details/9924377)

## 其他相关

### 用户冷启动

解决冷启动的方案<br>
1）**提供非个性化的推荐** 最简单的例子就是提供热门排行榜，可以给用户推荐热门排行榜，等到用户数据收集到一定的时候，再切换为个性化推荐。例如Netflix的研究也表明新用户在冷启动阶段确实是更倾向于热门排行榜的，老用户会更加需要长尾推荐

2）利用用户注册信息（人口统计学） 用户的注册信息主要分为3种：获取用户的注册信息；根据用户的注册信息对用户分类；给用户推荐他所属分类中用户喜欢的物品。

3）**选择合适的物品启动用户的兴趣** 用户在登录时对一些物品进行反馈，收集用户对这些物品的兴趣信息，然后给用户推荐那些和这些物品相似的物品。一般来说，能够用来启动用户兴趣的物品需要具有以下特点：

1. 比较热门，如果要让用户对物品进行反馈，前提是用户得知道这是什么东西；
2. 具有代表性和区分性，启动用户兴趣的物品不能是大众化或老少咸宜的，因为这样的物品对用户的兴趣没有区分性；
3. 启动物品集合需要有多样性，在冷启动时，我们不知道用户的兴趣，而用户兴趣的可能性非常多，为了匹配多样的兴趣，我们需要提供具有很高覆盖率的启动物品集合，这些物品能覆盖几乎所有主流的用户兴趣

4）利用物品的内容信息. 用来解决物品的冷启动问题，即如何将新加入的物品推荐给对它感兴趣的用户。物品冷启动问题在新闻网站等时效性很强的网站中非常重要，因为这些网站时时刻刻都有新物品加入，而且每个物品必须能够再第一时间展现给用户，否则经过一段时间后，物品的价值就大大降低了。

5）采用专家标注. 很多系统在建立的时候，既没有用户的行为数据，也没有充足的物品内容信息来计算物品相似度。这种情况下，很多系统都利用专家进行标注。

6）**利用用户在其他地方已经沉淀的数据进行冷启动**. 以QQ音乐举例：QQ音乐的猜你喜欢电台想要去猜测第一次使用QQ音乐的用户的口味偏好，一大优势是可以利用其它腾讯平台的数据，比如在QQ空间关注了谁，在腾讯微博关注了谁，更进一步，比如在腾讯视频刚刚看了一部动漫，那么如果QQ音乐推荐了这部动漫里的歌曲，用户会觉得很人性化。这就是利用用户在其它平台已有的数据。再比如今日头条：它是在用户通过新浪微博等社交网站登录之后，获取用户的关注列表，并且爬取用户最近参与互动的feed（转发/评论等），对其进行语义分析，从而获取用户的偏好。所以这种方法的**前提是，引导用户通过社交网络账号登录**，这样一方面可以降低注册成本提高转化率；另一方面可以获取用户的社交网络信息，解决冷启动问题。

7）**利用用户的手机等兴趣偏好进行冷启动**Android手机开放的比较高，所以在安装自己的app时，就可以顺路了解下手机上还安装了什么其他的app。比如一个用户安装了美丽说、蘑菇街、辣妈帮、大姨妈等应用，就可以判定这是女性了，更进一步还可以判定是备孕还是少女。目前读取用户安装的应用这部分功能除了app应用商店之外，一些新闻类、视频类的应用也在做，对于解决冷启动问题有很好的帮助。

### 反馈体系

**显示评分**：推荐系统推送给用户的明确的评分系统，比如为对商品的喜爱程度从1到10打分。

**隐式评分**：用户的一些特定行为可以解释为正向或是负向的反馈，例如用户购买某种商品可以解释为用户对于该类商品的正反馈，又或者是用户刷新信息流但是没有点击的动作可以解释为用户对于曝光的信息流都不感兴趣。可是隐式反馈的解释并不是总是正确的，例如用户想给父母买老年手机，搜索老年手机这个行为会被解释为用户对该型号手机感兴趣，并在用户实际购买了老年人手机之后持续一段时间的曝光，可是用户在完成购买行为之后就对老年人手机不感兴趣了，需要一个负反馈机制去抵消正反馈机制带来的“错误”。

### **AB Testing**

**A/A test**两个版本一样的AB Test； 目的在于检查实验平台的完整性，确保两个流量桶的用户分布是相同的，验证后续进行的AB test的有效性，如果AA 测试的结果达到统计显著，则AB Test的结果是不可信的。AB测试本身需要满足**同时性**(A,B两种策略同时上线)和**同质性**(A,B两个用户群体尽量性质相同)

**流量分组**<br>
首先流量分组需要满足**正交**和**互斥**，不同的流量层之间流量正交，同一层下流量互斥。**为什么要分层？**流量是有限的，但是实验数量是可以无限的，如果流量不分层，假设每个实验占据10%的流量，那最多同时进行10个实验，流量利用率极低，但是如果我们可以把参数互不相关，主要观察指标不同的两个业务拆开，例如广告层，推荐层等等，层与层之间共享100%流量，保持正交，层内保证流量互斥，则成倍的增加可同时运行的实验数。为了保证层间流量正交，需要在不同的层使用不同的随机分桶算法，上一层被分在同一个桶的流量，在下一层被分在每个桶的概率是相同的，也就是同一个流量在每个层被分为那个桶，是独立不相关的，下一层桶的流量可能来自上一层的每个流量桶，避免了“**流量偏置**”问题。**为什么层内流量要互斥？**如果同一层下两个实验流量有重叠，无法归因实验效果。注意**不要使用同一组用户反复试验**，长期反复使用同一组用户，会“驯化”用户，行为有偏，AB结果不准。

![abt](/assets/img/recommendation_system/abt.png)

在分层之上，还可能有一层结构“域”，从整体流量池中按某个维度拆分成不同的“流量区域”，例如分新老用户，新老用户有自己各自的实验，各自的分层，不同流量域流量总和为100%。

**实验设计**<br>
**单因素实验**，就是最常用的AB实验；**多因素实验**，同时修改两个或者多个变量，组合数为笛卡尔积，例如有2个实验，每个实验两个策略，组合方式有A1B1,A2B1,A1B2,A2B2四种，可以对比出两个实验组合的情况下，哪一种更好。

**数据分析**<br>
在进行AB实验之前，需要一段时间的空跑期也就是AA实验，用来验证流量分组的效果，而且AA两组没必要表现完全一致，如果AA两组有较为固定的“差异”，也是可以继续AB实验，只是最后AB的结果需要减去AA的“固有差异”。减去AA差异之后，利用统计学中的假设检验方法得出B组表现是否和A组有显著差异

**Reference**<br>[Athena-贝壳流量实验平台设计与实践](https://www.jianshu.com/p/79d31a72978f)<br>[推荐中AB实验最大的问题——流量偏置及解决方案：重叠实验架构](https://blog.csdn.net/zlb872551601/article/details/103757907)<br>[AB测试从应用到系统搭建](https://zhuanlan.zhihu.com/p/79690021)<br>[一文看懂AB测试](https://zhuanlan.zhihu.com/p/108916194)

### 假设检验

AB测试的结果是否显著的说明了B组和A组有差异，用户停留时长A组40min，B组41min是否说明B组效果好？有多大的信心说B组效果好？ 这些问题都要用假设检验来给出答案。

**假设检验的本质用通俗的话来说“假设骰子质地均与，我投了100次，只有1次6点，那么我有理由相信骰子有问题！”**，过程首先是提出原假设，叫**H0，零假设**(通常为=, >=, <=)。通常来说，原假设是统计者想要拒绝的假设，拒绝原假设的条件是实际数据“偶然”的落在拒绝域中，拒绝域通常为1%，5%，10%等，这个数是人为控制的能忍受的**弃真错误**($\alpha$错误，第一类错误，描述原假设为真但是抽样数据落在拒绝域的概率)的发生概率。另一个假设是**备择假设**H1(通常为 !=, >, <),涉及到第二类错误，**取伪错误**($\beta$错误，第二类错误)，即原假设是错误的，但是通过抽样数据的估计，接受了原假设，$\alpha + \beta$**不一定等于1** 

|                   | H0 is True     | H1 is True     |
| ----------------- | -------------- | -------------- |
| Reject H0         | Type 1 Error   | Right Decision |
| Fail to Reject H0 | Right Decision | Type 2 Error   |

第一类错误和第二类错误的关系可以用下图解释

![alpha-beta-error](/assets/img/recommendation_system/alpha-beat.png)

H0表示假设的分布，H1表示真实的分布，如果**检验统计量**落在了$\alpha$区域，有95%的置信度去拒绝原假设，H1成立，但同时有5%(显著性水平)的概率发生弃真错误。如果检验统计量落在$\beta$区域，没有足够理由拒绝H0，但有$\beta$的概率H0错误，H1成立，只不过$\beta$值很难衡量。

**检测方式**<br>
分为单侧检测和双侧检测，单侧检测分为左侧检测(a < b)和右侧检测(a > b)。

|          | H0               | H1             |
| -------- | ---------------- | -------------- |
| 双侧检测 | a = b            | a != b         |
| 单侧检测 | a >= b or a <= b | a > b or a < b |

![test](/assets/img/recommendation_system/test.jpg)

为了让问题不至于过复杂，**假设样本量大于30**，满足大样本条件

|            | 总体参数假设检验                                             |
| ---------- | ------------------------------------------------------------ |
| 假设形式   | 双侧： H0: $\mu=\mu0$; H1:$\mu\neq\mu0$ <br />左侧H0: $\mu\geq\mu0$; H1:$\mu<\mu0$<br />右侧H0: $\mu\leq\mu0$; H1:$\mu>\mu0$ |
| 检验统计量 | $z=\frac{\overline{x}-\mu_0}{\sigma/\sqrt{n}}$ 总体标准差已知；$z=\frac{\overline{x}-\mu_0}{s/\sqrt{n}}$总体标准差未知，用样本标准差拟合；n为样本量 |
| 拒绝域     | 双侧 \|Z\|>\|Z$\alpha/2$\| ; 左侧检验 Z<-Z$\alpha$; 右侧检验Z>Z$\alpha$ |
| 决策       | p-value < $\alpha$ 则拒绝原假设                              |

|            | 两个总体参数假设检验                                         |
| ---------- | ------------------------------------------------------------ |
| 假设形式   | 双侧H0: $\mu1-\mu2=0$ H1: $\mu1-\mu2\neq0$<br />左侧H0: $\mu1-\mu2\geq0$ H1: $\mu1-\mu2<0$<br />右侧H0: $\mu1-\mu2\leq0$ H1: $\mu1-\mu2>0$ |
| 检验统计量 | $t = \frac{(\overline{x1}-\overline{x2})-(\mu1-\mu2)}{\sqrt{s1^2/n1+s^2/n2}}$ |
| 拒绝域     | 双侧\|Z\|>\|Z$\alpha/2$\|; 左侧检验Z<-Z$\alpha$; 右侧检验Z>Z$\alpha$ |
| 决策       | p-value < $\alpha$ 则拒绝原假设                              |

**Reference**<br>[第一类错误和第二类错误的关系是什么？](https://www.zhihu.com/question/20993864)<br>[假设检验——这一篇文章就够了](https://zhuanlan.zhihu.com/p/86178674)<br>[假设检验总结以及如何用python进行假设检验（scipy）](https://www.cnblogs.com/HuZihu/p/11442833.html)

### Bandit算法

源自赌博学上的问题，一个赌徒，面对多个老虎机，每个老虎机的期望收益不同，但事先不知道具体那台老虎机赢钱的概率最大，手上有多个硬币，每次摇动老虎机的机械臂需要花费一个硬币，那如何选择老虎机来使期望收益最大呢？这就是**多臂老虎机问题(Multi-Armed Bandit Problem)** 

首先考虑两个极端情况，所有机器我都一视同仁，都投入相同的资源，称为**explore**(探索)，另一个是我每个机器试一次，然后剩下的钱全部投入收益最高的那台机器，称为**exploit**(利用)。很明显，这两种方法在逻辑上都不是最优化的方法，都存在明显漏洞。explore方法虽然尽可能的了解到了每台机器的预期收益，但并没有利用高价值老虎机，exploit方法在试验次数过少的前提下过早的采用“贪婪”思想，当前收益最高的机器是全局期望收益最高的机器的可能性不大，当然运气好除外。那很明显，优化的方向就在于平衡explore和exploit过程。首先花一定的成本去了解每台机器的预期收益，该预期收益在可控成本的范围内尽可能的接近真实预期收益，用样本分布区拟合真实分布，然后再利用当前样本收益最高的机器去获得最高的收益。

如何平衡EE问题就是Bandit算法的核心了，Bandit算法需要量化一个核心问题：**错误的选择有多少遗憾**？

$$
R_t = \sum_{i=1}^{T}(w_{opt}-w_{B(i)})\\
=TW^* - \sum_{i-1}^{T}w_{B(i)}\\
$$

其中，B(i)是第i次试验被选中臂的期望收益，w\*是最优的收益。不同的bandit算法之间的好坏就是通过累积遗憾的大小和增长速度来衡量，这里描述的收益为**伯努利收益，只有0,1两种取值**，常见Context-Free Bandit算法如下：

**Thompson Sampling**<br>
1. 用beta分布去描述每一个机器的收益分布,初始beta分布为Beta(1,1)
2. 每台老虎机维护各自beta分布，在探索和利用过程中，更新分布参数Beta(1+wins, 1+lose)
3. 选择机器的原则是每个机器依据现有分布产生一个随机数，选择随机数最大机器

首先随机数的产生保证当前收益最大的机器并不会一直被利用，其他机器也有被利用的可能，同时当前收益高的机器产生大随机数的可能性较大，也保证一定的“贪婪”

**UCB **(Upper Confidence Bound )<br>
置信区间上界算法，步骤如下<br>
1. 每一台机器试一次
2. 依据结果计算分数值

$$
\overline{x}_j(t)+ \sqrt{\frac{2lnt}{T_{j,t}}} 
$$

其中$x$是当前机器的收益均值$T_{j,t}$是当前机器的被试次数，$t$是总实验次数，每次选择都选择当前分数最高的机器实验，更新分数值

UCB公式的前半部分衡量是收益的平均值，后半部分衡量了波动性，当前被试次数越少波动性越大，分数总和可以看做mean+std（置信区间）衡量了最大的潜在收益，如果一个机器mean很小，但是没有被试几次，std很大，那么潜在收益区间很大，倾向于被试，随着被试次数的上升，std收窄，渐渐倾向于mean大的机器，每次总是选择置信区间上界最大的那个机器。其实也可以选择置信区间下界最大，这样很保守，公式如下

$$
\overline{x}_j(t) - \sqrt{\frac{2lnt}{T_{j,t}}}
$$

从公式上来看，一个机器被试的次数越少，分数越低，平均收益越高，分数越高，也就是**倾向于选择平均收益高，而且被试了很多次，收益稳定的那台机器**。

**Epsilon-Greedy**<br>
比较朴素的方法，引入一个$\epsilon$ ,以$\epsilon$的概率去随机选择一个老虎机，否则选择目前平均收益最大的那台机器

**朴素Bandit算法**<br>
真朴素算法，每个机器尝试若干次之后，一直选择收益均值最大的机器。

**Bandit算法与推荐系统的联系**<br>
推荐领域EE问题客观存在，Bandit算法是EE问题的一种量化解决方式。现阶段，工业界更多在深耕Exploitation，把用户的历史行为钻研透，尽可能用历史行为画出用户的精准画像，一定程度上忽略了用户的潜在兴趣，积累够足够的行为后，用户很难被展示非历史兴趣的内容，导致推荐系统“收敛”，用户反复被展示同一pattern的信息，行为肯定也是基于这一pattern的反馈，推荐系统接收这些行为，形成正反馈闭环，用户行为被“框死”，导致用户偶尔非常规的探索欲得不到满足，损害长期受益。但Exploration 的风险同样不能忽略，同一时间用户能接受到的内容有限，探索的内容通常受益很低甚至损失收益，固然主体是利于用户历史的推荐，一小部分展示位留给探索。但即便如此，很多产品只有很少甚至没有探索的逻辑，原因如下<br>

- 通常互联网产品的生命周期很短，不需要探索逻辑<br>
- 用户碎片化的使用习惯，探索收益不大<br>
- 同质化产品多，不能时刻留住用户，面临用户流失<br>

添加探索逻辑需要注意

- 提高产品生命周期，让探索发挥作用，也就是不能为了Exploration，Exploitation没做好
- 探索时尽量不损害用户体验,步子不要迈太大

**Reference**<br>[Beta分布]([https://baike.baidu.com/item/%E8%B4%9D%E5%A1%94%E5%88%86%E5%B8%83/8994021?fr=aladdin](https://baike.baidu.com/item/贝塔分布/8994021?fr=aladdin))<br>[大白话解析模拟退火算法](https://www.cnblogs.com/heaad/archive/2010/12/20/1911614.html)<br>[【总结】Bandit算法与推荐系统](https://blog.csdn.net/dengxing1234/article/details/73188731)<br>[广告场景中的Explore与Exploit](https://zhuanlan.zhihu.com/p/136638444)<br>[The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)