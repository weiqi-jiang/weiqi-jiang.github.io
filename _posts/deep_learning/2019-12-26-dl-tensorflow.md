---
layout: post
title: TF(TensorFlow) Boys
category: DeepLearning
tags: tensorflow
description: TensorFlow 用法
---



## 安装

Tensorflow的安装...一言难尽，坑真的多

**情况一：python3.7 + tensorflow2.1.0**(lastest)

Error:  msvcp140.dll丢失

Solution:  

下载   [Microsoft Visual C++ 2015 Redistributable Package](https://www.microsoft.com/en-us/download/details.aspx?id=48145) ,  本人下载完之后安装报错，原因是有更高级版本，不允许安装低级版本。而且检查了系统路径下是存在目标dll文件的，推测是版本问题。

**情况二：python3.6.8 + tensor2.1.0**(lastest)

Error：如果直接用pip install tensorflow 报错，原因是找不到合适的tensorflow版本

Solution：手动找到[tensorflow2.1.0](https://pypi.org/project/tensorflow/#modal-close) 64位python3.6版本, 安装成功，但是当import tensorflow 的时候会报错

ImportError: DLL load failed: 找不到指定的模块；Failed to load the native TensorFlow runtime.

**情况三：python3.6.8 + tensor1.4 + keras（latest）**

Error：安装成功，可以正常import，但是在pip install keras之后 如果import keras，会报ImportError: cannot import name 'tf_utils'错误

Solution：tensorflow 和keras的兼容问题，把keras版本降低至2.0.8解决



----------------------------------------2020.3.30更新---------------------------------------------

发现新坑，如果本机上有多个python, 例如anaconda一个python，自己另外安装了一个python，pip install xxx的时候容易出错，版本对不上，需要先确定默认的python和pip版本，然后在安装，不容易出错



### 可能用到的指令

```
# 卸载tensorflow
pip uninstall tensorflow

# 安装指定版本的tensorflow
pip install tensorflow==1.7.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.4.0

# 升级pip
python -m pip install --upgrade pip

# 查看已经安装的module
pip list

# 搜索module
pip search tensorflow

# 查看默认的python和pip
# 修改默认python 和pip的方法很简单，在环境变量中修改路径的前后顺序就可以，默认版本的路径前提，非默认版本后放
which python
where python
which pip
where pip
```



**Reference**

[tensorflow，keras，python版本对照表](https://docs.floydhub.com/guides/environments/)



## 背景知识

### 理论
深度模型强调两个点： ***多层，非线性***

**非线性**：多层线性模型等价于一层线性模型，线性模型解决的问题很有限，所有强调非线性

**多层**： 这里的多层并不是指很多个hidden layer，实际上一层hidden layer就足够了，如果没有hidden layer 就是perceptron,  perceptron是不能解决异或问题的

### 底层
**Protocol Buffer**

类似于JSON 和XML用于处理结构化数据，但是不同于XML,JSON，Protocol Buffer序列化之后的数据不是可读的字符串而是二进制流，并且**Protocol Buffer需要先定义数据的格式(schema)**。由多个message构成，每个message代表一类结构化的数据类似于编程语言中的class。message 中的变量有optional，repeated，required 三种修饰

```
message user{
optional string name = 1;
required int32  id = 2 ;
repeated string email = 3;

}
message item {
 required string name = 1;
 optional string categories = 2;
}
```

其中 变量后面的数字表示编号，在进行编码解码的时候用来保证顺序不会错位。

**TFRecord**

TFRecord 内部有一系列的Example， example是protocolbuf 协议下的消息体，protocol buffer 和json差不多

```
Example{
Features features = 1
}

Features{
map<string, Feature> feature = 1  # map feature name to feature
}

Feature{
  oneof kind{
  ByteList bytes_list = 1;
  FloatList float_list = 2;
  Int64List int64_list = 3;
  }
}

所以Example 就是一系列map 每个map中把featurename map 到feature 
key-value 都是列表形式
```

下图是Example 的一个示例，把一张图片分为“image”“label”两个维度来存储

![img](/assets/img/deeplearning/tensorflow/tfrecord-example.png)

生成TFRecord之后使用tf.parse_single_example() 或者parse_example() API 去读取TFRecord

```
tf.parse_single_example{
  serialized;
  features;
  name = None;
  examplename = None
}

serialized: 序列化之后的tensor
features： 一个map 把featurename map to FixedLenFeatures/VarLenFeatures/SparseTensor 类型中的一个
```

 

## 入门

### graph

> tensorflow每一个计算都是都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系

系统自动维护一个默认的计算图, tf.get_default_graph 得到这个图。通过tf.Graph() 创建新的图。不同图上的张量和运算都不共享. TF1.0 采用的是静态计算图的形式，也就是程序首先构造神经网络的结构，再执行数值计算，TF2.0中支持动态计算图，允许程序安装编写的顺序执行，这样调试更加容易，在编写一个算子之后，该算子被动态加入隐含的默认计算图中，进行数值结算，可以即刻打印结果，不用开启session显示运行计算图之后才能得到实质的运算结果。

```python
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x,y],separator=" ")
print(z)# 可以直接得到结果
```

在TF2.0中默认采用的是动态计算图，但是理论上动态计算图的计算效率比静态计算图低，使用AutoGraph工具将python函数转化为图表示代码,只需要用@tf.function 装饰python函数即可。这样在TF1.0中定义图结构，在session中运行数值计算，在TF2.0中变成了定义函数，调用函数的过程，和原生python的开发流程相同。

```python
# tf.function装饰器装饰过之后，eager excution不适用于该函数，可适当提高性能
@tf.function
def string_join(x,y):
	z = tf.strings.join([x,y],separator=" ")
	return z
result = string_join(tf.constant("hello"),tf.constant("world"))

```

### session

**tensorflow会自动生成默认的计算图，但是不会生成默认的session**，如果没有指定模型session，with sess.as_default(),  tf.Tensor.eval()需要指定session才能得到tensor的值

config = tf.ConfigProto() 来配合生成的会话，最常用allow_soft_placement=True,这个参数允许GPU在特定情况下可以在CPU上运行，而不是报错。

### collection

```python
tf.GraphKeys.VARIABLES   #所有变量
#通过tf.global_varibles() 获取

tf.GraphKeys.TRAINABLE_VARIABLES  #可学习的变量
#通过tf.trainable_variables() 获得

tf.GraphKeys.SUMMARIES  #日志生成相关的张量

tf.GraphKeys.QUEUE_RUNNERS #处理输入的QueueRunner

tf.GraphKeys.MOVING_AVERAGE_VARIABLES  #所有计算了滑动平均的变量
```

### tensor

> 张量在功能上相当于多维数组，但是实现并不是直接采用多维数组的形式，只是对tensorflow中运算结果的引用，并没有保存真正的数字

通过调用tf.Tensor.eval(tensor,session = sess) 可以得到张量的具体值,或者tensor.eval(session = sess)。张量主要保存三个属性： **名字，维度和类型**,其中标量是0维张量，向量1维张量，矩阵2维张量，图像3维张量，视频4维张量。

**常量张量**， 常量值不能改变，修改会重新创建新的常量张量

```python
import tensorflow as tf

i = tf.constant(1) #int32类型常量张量
l = tf.constant(1, dtype=tf.int64) # 指定数据类型为int32
f = tf.constant(1.23)# float32 类型
d = tf.constant(1.23,dtype=tf.double)# double
s = tf.constant('hello world') # string 
b = tf.constant(True)# boolean
r = tf.range(1, 10, delta=2) # tf.range(start, limit, delta) 左闭右开
lin = tf.linspace(0, 6, 100) # tf.linspace(start, stop, num) 生成0-6之间均匀分布的100个点，双闭区间
z = tf.zeros([3,3]) # 一个3*3的全是0的二维矩阵
o = tf.ones([3,3]) # 一个3*3全是1的二维矩阵
new_z = tf.zeros_like(z,dtype=tf.float32)# 按照z张量的维度生成全是0的矩阵张量
fi = tf.fill([3,2], 5) # 用5填充3*2维度的矩阵

print(tf.rank(i)) # 打印张量的维度
new_i = tf.cast(i,dtype=tf.int64) #改变数据类型
tensor2numpy = l.numpy() # 转换为numpy array
print(d.shape)# 打印张量尺寸
```

**变量张量**

```python
import tensorflow as tf

# 主要是两个方式 tf.Variable tf.get_variable
v = tf.Variable([1,2], name='var') 
v1 = tf.Variable(tf.truncated_normal([d1,d2], stddev=0.1)) #用随机数生成器来赋值

# 首先需要定义变量namespace, get_variable 如果在命名空间下没有该name的变量，则新建，如果有，则直接获得
# 这种方式可以方便的进行变量间的共享，不需要把用到的变量都作为参数传到函数中,但是必须将reuse=True
with tf.variable_scope('foo'):
  v2 = tf.get_variable('v',[1], initializer=tf.constant_initializer(1.0))

# 当 reuse=True 时，直接获取已经创建好的变量，如果没有，报错，reuse=None or False的时候
# 当嵌套variable_scope  内层没有指定reuse 时，取值和外层保持一致，如果是最外层没有指定，默认是false
# 创建新变量，如果变量已经存在，报错
with tf.variable_scope('foo', reuse=True):
  v2 = tf.get_variable('v',[1])

# 如果其他地方要用到之前的变量
with tf.variable_scope('',reuse=True):
  v3 = tf.get_variable('foo/v2',[1])
```

**随机数生成函数**

```python
# 更多的例子见文档
tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None,name=None)
tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None,name=None) # 左闭右开
tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None) #剔除掉两倍标准差以外的数据，重新生成

# TF version 1.0
tf.random_normal   #正太分布；   参数：mean,stddev,dtype,seed,name
tf.truncated_normal #受限正太分布，如果偏移均值两个标准差，重新随机 mean,stddev,dtype,seed,name
tf.random_uniform   #均匀分布； 参数： maxval，minval,dtype,seed,name
tf.random_gamma   #gamma分布  参数，alpha,beta,dtype
```

**特殊矩阵**

```python
tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.dtypes.float32, name=None)# 单位矩阵
tf.linalg.diag(
    diagonal, name='diag', k=0, num_rows=-1, num_cols=-1, padding_value=0,
    align='RIGHT_LEFT'
)# 对角阵 tf.linalg.diag([1,2,3]) 对角元素分别是1,2,3，其他用padding_value填充

```

**索引/切片**

```python
# 张量基本切片方法和numpy 的方法差不多，更进阶的方法用的时候参考文档即可
t = tf.random.uniform([5,5],minval=0,maxval=10,dtype=tf.int32)
t[0]# 访问第2行
t[1][3] # 第2行第4个元素
t[1:3, :] #第2到4行

''' tf.where(condition, x=None, y=None, name=None)如果x,y不提供，返回condition为true的index
返回一个tensor 2*1维的tensor，2表示有两个元素满足要求，1维是因为c张量是一维的，如果c维度变化，结果维度也变化，值是1,2表示index。 如果x，y提供，则当condition满足的时候取x对应index位置的元素，否则取y对应位置的元素
'''
c = tf.constant([1,2,3])
tf.where(c>=2) # 返回满足condition的indices
tf.where(c>=2, tf.fill(c.shape, tf.nan), c) # 除了满足条件的位置为c中对应元素，其他地方全为nan的矩阵
```

**维度变换**

```python
# tf.reshape 不会改变元素的存储顺序，快速且可逆
a = tf.random.uniform([2,3,4])
tf.reshape(a, [2,12])

# tf.squeeze 如果张量在一个维度上只有一个元素，则消除这个维度
a = tf.random.uniform([1,2,3,4])
b = tf.squeeze(a) # TensorShape(2,3,4)

# tf.expand_dims(input, axis, name=None) 在axis指定的地方插入一个长度为1的维度
c = tf.expand_dims(b, axis=1) #TensorShape(2,1,3,4)

# tf.transpose 转置 ,perm表示把原来的维度按照什么顺序排列，数字表示原来tensor维度的index
tf.transpose(c, perm=[3,1,0,2]) # TensorShape(4,1,2,3)
```

**合并/分割**

```python
# tf.concat  不会新增维度个数，二维concat还是二维但是张量shape会变
a = tf.constant([[1,2], [3,4], [1,2]])
b= tf.constant([[5,6], [7,8], [5,6]])
c = tf.concat([a,b], axis=0) #纵向拼接 也就是新增两行 tensorshape(6,2)
d = tf.concat([a,b], axis=1) # 横向拼接，也就是新增两列 tensorshape(3,4)

# tf.stack 会新增维度 axis default=0 原来的tensor维度是(A,B,C)如果axis=1则新维度(A,N,B,C)依赖类推
tf.stack([a,b]) # tensorshape)(2,3,2)

# tf.split  tf.concat的逆操作
tf.split(c, 3,axis=0) # 横着切成3份 平均分割
tf.split(c,[1,2,3], axis=0) # 指定每份分割的数量，第一part1个sample，第二part2个类推
```

**Reference**<br>[TensorFlow 文档v2.3.0](https://www.tensorflow.org/api_docs/python/tf/random/normal)

### operator

**数值运算**

```python
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])
c = tf.constant([[9, 10],[11, 12]])
# 矩阵点乘(内积),对应位置相乘
a*b 

# 矩阵相乘,行乘列相加
tf.matmul(a, b)
a@b

# 设a中元素是m, b中对应位置元素是n，则结果是m^n
a**b

# 对应位置相加
a+b 

# 各元素平方/开方，开方运算需要至少是float类型，不能是int
a**2 
tf.math.sqrt(a) 

# a中元素除以b中元素
a/b

# 整数除
a//b

# 累加
a+b+c 
tf.add_n([a, b, c])

# maxium, minium 两个张量的类型需一致
tf.math.maxium(a,b)
tf.math.minium(a,b)
''' maxium
Array[
	[5, 6],
	[7, 8]
	]
'''

# 幅值裁剪, 把超过幅值的元素值用上下幅值替代
x = tf.constant([1,2,3,4,5,6,7])
tf.clip_by_value(x, clip_value_min=2, clip_value_max=6) #[2, 2, 3, 4, 5, 6, 6]
# 变换公式：t*clip_norm/l2norm(t)， 
tf.clip_by_norm(x, clip_norm=5) 

```

**逻辑运算**

```python
(a>=2)
'''
Array[
 [False, True],
 [True, True]
     ]
'''

#需要a，b有同样的尺寸
(a>=2)|(b<=6)


```

**向量运算**

```python
x = tf.range(1,10)

# 按照规则将张量聚合
tf.math.reduce_sum(x)
tf.math.reduce_prod(x)
tf.math.reduce_min(x)
tf.math.reduce_mean(x)

# 按照维度reduce
y = tf.reshape(x,(3,3))
tf.math.reduce_sum(y, axis=1, keepdims=True) #array[[6],[15],[24]]
tf.reduce_sum(y, axis=0, keepdims=True) # array[[12,15,18]] 如果不指定keepdims，不保留原来的维度，返回一个array张量

# bool类型reduce
b = tf.constant([True, False, True])
tf.math.reduce_all(b) # and关系 返回False tensor，tf.print(tf.math.reduce_all(b))返回0
tf.math.reduce_any(b)

# 累加，累乘
# [a,b, c] -> [a, a+b, a+b+c] 如果设置exclusive参数为True ->[0,a,a+b],设reverse为True ->[a+b+c, b+c, c]
tf.cumsum(x)
tf.cumprod(x) # 同cumsum，加法变乘法

# 最大最小值索引
ma = tf.math.argmax(x)
mi = tf.math.argmin(x)

# top_k 如果是一维，返回对应的k个最大值和索引，如果是多维，计算每一行的top_k,和索引
value, index = tf.math.top_k(x)
```

**矩阵运算**

```python
# 矩阵点乘，叉乘在数值运算的模块中提到了
a = tf.constant([1,2],[3,4])
tf.transpose(a) # [[1,3],[2,4]]

# 矩阵的逆 ,必须是float或者double类型
tf.linalg.inv(a)

# 矩阵的迹 trace，n*n主对角线上元素值的和
tf.linalg.trace(a)

# 矩阵的范数,默认l2范数
tf.norm(a, ord=2)

# 特征值
tf.linalg.eigvals(a)

# 行列式
tf.linalg.det(a)

# qr分解
q,r = tf.linalg.qr(a)
q@r

# svd分解
s,u,v = tf.linalg.svd(a)
u@tf.linalg.diag(s)@tf.transpose(v)
```

**广播机制**

- 让所有数据都向其中形状最长的数据看齐
- 输出数组的形状是输入数组形状在各个维度上的最大值
- 如果输入数组的某个维度和输出数组的对应维度的长度相同或者长度为1时，可以计算，否则报错
- 当输入数组的某个维度的长度为1时，沿着此维度运算时都用此维度上的第一组值

```python
a = tf.constant([[1,2],[3,4]])
b = tf.constant([10,10])
a+b  # [[11,12],[13,14]]
```

### dataset

```python
""" 通常流程
数据管道本质是一个ETL过程Extract,Transform, Load
1. 首先提取数据Extract，可以从from_tensor_slice, from_generator, TextLineDataset
2. 再Transform, 通常预先定义预处理函数，调用dataset.map对每个元素应用预处理函数
3. 最后调用repeat,batch,prefetch几个常用的方法完成最后的“load”

"""


############ 从tensor，numpy，dataframe ###############
dataset1 = tf.data.Dataset.from_tensor_slices(data)
dataset2 = tf.data.Dataset.from_tensors(data) # 生成只有一个元素的dataset

############# 从文件 ###################
BATCH_SIZE = 64

# 返回一个由文件名字符串构成的dataset
file_dataset = tf.data.Dataset.list_files("/path/*.csv")

# 多进程把不同来源的数据交错在一起，其中interleave的第一个参数是map_func。
dataset = file_dataset.interleave(
	lambda file: tf.data.TextLineDataset(file).skip(1)
	)

# 读取出来的数据经过预处理，第二个参数表示并行处理的元素个数
dataset.map(preprocess_func, num_parallel_calls=10)

# 打乱,buffer_size需要大于等于dataset数据量,详情见文档
dataset.shuffle(buffer_size=len(dataset))

# 重复，如果不指定repeat参数，则无限重复
dataset = dataset.repeat()

# 设置batch 大小
dataset = dataset.batch(BATCH_SIZE)

# prefetch,文档中提到绝大多数dataset应该以prefetch方法调用结尾，使得当前元素在处理时，下一个元素能同时被准备，空间换时间
dataset = dataset.prefetch(buffer_size=5)

# 当然可以把prefetch，batch，repeat一起写
ds = dataset.repeat().bacth(64).prefetch(5)
```

**Reference**<br>[tf.data.Dataset.shuffle(buffer_size)中buffer_size的理解](https://juejin.im/post/6844903666378342407)<br>[一文上手最新Tensorflow2.0系列(三)  “tf.data”API 使用](https://www.jianshu.com/p/e8ae78bef371)

### feature_column

常常用于对结构化数据进行特征工程，将常用的连续值分桶，类别特征one-hot编码等封装起来，直接指明某某字段是什么类型的特征，不用显式写特征工程代码，Tensorflow自动完成，并喂给模型。

```python
import numpy as np 
import pandas as pd
import tensorflow as tf	
from tensorflow import feature_column
from tensorflow.keras import layers

############# numerical feature ###################
# numerical columns，这一步仅仅是指定了age特征是一个数值特征，并没有传具体的data进去
# 方法返回feature_column类型对象
age = feature_column.numeric_column("age")

# bucketized columns, 输入是一个numeric_column,左闭右开，输出的特征数=len(boundaries)+1
age = feature_column.bucketized_column(age,boundaries=[30,50,80])

############# categorical feature #################
# 所有categorical feature 最后都要经过indicator column转成Dense column才能喂给模型/除了embedding column
# categorical column with vocabulary list, 本质就是one-hot编码
grade = feature_column.categorical_column_with_vocabulary_list(
      'grade', vocabulary_list=['poor', 'average', 'good'])
grade = feature_column.indicator_column(grade)

# embedding column 如果类型太多，onehot编码后会过于稀疏，而且维度过大，可以用embedding column降维
# embedding 维度是个超参，embedding之后数值是浮点数不是0,1 boolean,lookup table 其实在训练的时候同步更新的，而不是提前就知道的
movie_cate = feature_column.categorical_column_with_vocabulary_list("point",df['point'].unique())
movie_embedding = feature_column.embedding_column(movie_cate, dimensions=4)

# hash bucket,适合类别过多的特征，对输入计算hash值，hash值除hash_bucket_size_取余，余数one-hot编码。缺点是不同的输入可能落到相同的桶
movie_hash = feature_column.categorical_column_with_hash_bucket('point', hash_bucket_size=4)
movie_hash = feature_column.indicator_column(movie_hash)

# crossed_column 输入可以任意categorical column,除了hash_bucket
# 不会生成full table of all possible combination，因为有可能很大，hash_bucket_size设定大小
gender_cate = tf.feature_column.categorical_column_with_vocabulary_list(
          key='gender',vocabulary_list=["male","female"])

crossed_feature = tf.feature_column.indicator_column(
    tf.feature_column.crossed_column([gender_cate, grade],hash_bucket_size=15))

feature_columns.append(crossed_feature)
```

上面代码基本涵盖了常用的几种特征列, 但是没有提到特征列处理好之后怎么喂给模型

```python
feature_columns = []

feature_columns.extend([age, grade, movie_embedding, movie_hash, gender_cate])

model = tf.keras.Sequential([
  # 需要把特征列放在DenseFeature中,相当于feature_column和keras模型之间的桥梁
  layers.DenseFeatures(feature_columns),
  layers.Dense(64, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

```

**Reference**<br>[Demonstration of TensorFlow Feature Columns (tf.feature_column)](https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4)<br>[Train tf.keras model in TensorFlow 2.0 using feature coulmn](https://medium.com/ml-book/train-tf-keras-model-using-feature-coulmn-8de12e65ddec)

### Activation

```python
tf.nn.sigmoid
tf.nn.softmax
tf.nn.tanh
tf.nn.relu
tf.nn.leaky_relu
tf.nn.elu
tf.nn.selu
tf.nn.swish

# 使用激活函数
model = models.Sequential()
# 第一种方式, 也可以写成activation='relu'
model.add(layers.Dense(32, input_shape=(None,16),activation=tf.nn.relu ))
# 第二种方式
model.add(layer.Dense(32))
model.add(layers.Activation(tf.nn.softmax))
```

### Layers

```python
# 常用
layers.Dense # 全连接层
layers.Activation
layers.Dropout
layers.BatchNormalization
layers.Input
layers.DenseFeature
layers.Embedding
layers.LSTM
layers.RNN
layers.Attention
```

### Loss&Metrics&Optimizer

```python
######### loss func 常用############
mean_squared_error #回归
mean_absolute_error
Huber
binary_crossentropy # 二分类
categorical_crossentropy # 多分类
hinge # svm
cosine_similarity # 余弦相似度

############ metrics 常用 ############
MeanSquaredError
MeanAbsoluteError
RootMeanSquaredError
Accuracy
Recall
AUC

############# optimizer 常用 ############
SGD
Adam
Nadam
Adadelta
Adagrad
RMSprop
```

### persistence

```
# 保存模型
# 模型保存有三个文件
# 后缀是.meta 保存计算图的结构
# 后缀.ckpt 保存每一个变量的取值
# 一个checkpoint文本文件，保存目录下所有的模型文件列表
with tf.Session() as sess:
  saver = tf.train.Saver()
  saver.save(sess,'path/to/model/checkpoint')
# 加载
# 这种方式需要提前复现一个计算图，定义计算图上的所有计算
with tf.Session() as sess:
  saver = tf.train.Saver()
  saver.restore(sess,'path/to/model/checkpoint')
  sess.run(result)
# 这种加载方式不需要复现计算图，直接加载
saver = tf.train.import_meta_graph('path/to/model/model.ckpt.meta')
with tf.Session() as sess:
  saver.restore(sess, '/path/to/model.ckpt')
  sess.run(tf.get_default_graph().get_tensor_by_name('add:0'))
```

### IO stream

使用queue 读取硬盘中的数据

```
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # 初始化一个reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       }) #解析数据
```

使用Dataset API(目前基于queue的方法在新版中已经移除，推荐使用dataset api)

[tf.data.Dataset 文档](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

```python
import tensorflow as tf
from tf.data.Dataset import *

# 数据集放在内存中 使用tf.data.Dataset.from_tensor_slices or tf.data.Dataset.from_tensors()
dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
  try:
    while True:
      print(sess.run(one_element)) # 打印1 到6
  except tf.errors.OutOfRangeError:
    print('end!')

# 从csv 文件中读取数据生成dataset,第一种方法是data.experimental.make_csv_dataset()函数，
# 返回一个dict结构的dataset元素列表，一个feature_name对应一个tensor
# 所以可以用dict访问方式 访问指定feature 或者label对应的tensor
def read_dataset_from_csv(filename):
    dataset = tf.data.experimental.make_csv_dataset(
        filename, batch_size = BATCH_SIZE, column_defaults=[0.0]*10, num_epochs=20, shuffle=True)
    iterator = dataset.maek_one_shot_iterator()
    return iterator.get_next()

# 还有一种是tf.data.CsvDataset()
dataset = tf.data.CsvDataset(filenames, record_defaults, header = ...)

# 从tfrecord 文件构建dataset, filename 可以是string 也可以是list of string
dataset = tf.data.TFRecordDataset(filename)

# 从文本文件构建dataset, filename 可以是string 也可以是list of string， 默认每次读取每个文件的一行
dataset = tf.data.TextLineDataset(filename).skip(1).filter(lambda line: conditions...)

# dataset 元素变换
dataset1 = dataset1.map(lambda x: ...)
dataset2 = dataset2.flat_map(lambda x,y: ...)
dataset3 = dataset3.filter(lambda x,(y,z): ...)

# dataset 聚合, 重复， 打乱
batch = dataset.batch(BATCH_SIZE)
dataset = dataset.repeat(n) # 如果不指定n 无限重复
dataset = dataset.shuffle(buffer_size = bs)
```

Dataset 类是***相同元素的有序列表\*，**元素类型很多，可以是字符串，图片，tuple，或是dict， 一个元素有多个tf.Tensor对象，对象被称为组件，可以为元素中各个组件命名，形式为dict， {‘name1’: tensor1, 'name2': tensor2} 。从dataset中把元素取出来的方法是通过迭代Iterator。iterator.get_next()返回的只是一个tensor并不是真实的值，只有通过sess.run() 才能真正的得到一个值, 每次eval tensor之后 迭代器才会进入下一个状态。如果dataset 元素读取完毕，再尝试sess.run(), 会抛出tf.error.OutOfRangeError错误。try, except机制去判断是否读取结束。

### SparseTensor Class

```
# tensorflow 用三个独立的稠密张量来表示SparseTensor
# indices, value, dense_shape
# indices 表示稀疏表示中非零项的index 例如indices = [[1,2],[3,4]]表示索引为[1,2][3,4]的元素值为非零
# values 和indices 的维度一样，对应每个非零元素的元素值
# dense_shape 指定稠密张量的形状
```

sparse_reorder 接受一个SparseTensor 类 返回一个维度不变的类 但是index和value 重新按照row-major规则排序好。

```
tf.sparse_reorder(
 sp_input,name = None
)
```

tf.nn.embedding_lookup_sparse 计算embedding

按照sp_id 找params对应行，乘上weights ， 按照strategy 进行reduction 最后组成一个tensor返回

```
tf.nn.embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy='mod',
    name=None,
    combiner=None,
    max_norm=None
)

# params ： embedding matrix
# sp_ids: sparseTensor 类型 需要使用 tf.sparse_reorder(sp_ids) 提前把indices 化为规范的row-major 
# sp_weights： 可以使具有float /double weight的SparseTensor 或者是none sp_weights 必须和sp_ids 完全相同的shape 和 indices
# partition_strategy 指定分割模式，支持div 和mod 默认mod
# combiner 指定reduction的操作符 目前支持“mean”,“sqrtn”和“sum”.“sum”计算每行的 embedding 结果的加权和.“mean”是加权和除以总 weight.“sqrtn”是加权和除以 weight 平方和的平方根. 
```

 

 



## 示例

**TF1.0简单实例**

```python

# 一个简单的正向传播过程
import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

# constant 注意有两个[] 不然会报错
x = tf.constant([[0.7,0.9]])
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
'''
这个随机生成输入，一层hidden layer的DNN网络
构建深度网络的一般步骤：
1.定义nerual network结构，定义前向传播的计算结果
2.定义损失函数，选择反向传播优化算子
3.生成会话，迭代运行优化算法
'''

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

### generate random input 
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]


###network
# input
x = tf.placeholder(tf.float32, shape = [None,2],name = 'input')
x_label = tf.placeholder(tf.float32,shape = [None,1],name = 'label')

# network
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1, seed = 1))
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
# output
predict = tf.sigmoid(y)

# training 
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_label, logits=predict))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    STEP = 5000
    for i in range(STEP):
        start = (i*batch_size)%batch_size
        end = min(dataset_size,start+batch_size)
        sess.run(train_step, feed_dict = {x:X[start:end],x_label:Y[start:end]})
    
        if i %1000  == 0:
            cross_entropy = sess.run(loss,feed_dict = {x:X,x_label:Y})
            print('Step %d, Loss %g' %(i,cross_entropy))
```

 




## Distributed TF

### 单机多GPU的工作模式：

由CPU 负责把batch发给多GPU，多个GPU 负责计算梯度更新，等待所有GPU 运算完毕，梯度更新数据发送给CPU，CPU 计算平均更新梯度，进行梯度更新，接着发送新的batch给多个GPU，时间消耗取决于最慢的GPU 和CPU,GPU 通信时间。

 

### 多机多GPU 模式：

当数据量急剧增大的时候，参数更新的速度就成了一个大问题，于是单机模式下单CPU 进行参数更新的模式就行不通了，引入Parameter Server (PS)的概念。组成集群进行梯度更新

Tensorflow 分布式给予gRPC通信框架(google Remote Proceduce Call), 简单理解就是把参数进行打包上传给服务器，服务器端进行运算，运算完之后把运行结果打包传回客户端。

 

### 分布式并行策略：

模型并行: 将模型的不同部分并行分发到不同的worker上训练，但是模型层与层之间是存在串行关系的，所以一般只有在不存在串行的部分进行模型并行

数据并行： worker使用的是同一个graph，但是数据是分块发给不同worker的，分别进行计算，参数更新模式有同步和异步之分。

### 参数更新

worker和ps之间的通信都由worker发出pull和push请求，什么时候发出请求是由scheduler调度。

**同步参数更新：**

workers 运算得到梯度更新参数，等到所有worker跑完一个batchsize，把参数更新信息发送给ps，计算平均梯度，ps进行梯度更新，然后把更新后的参数值传回worker，循环进行。

优点： 虽然单个worker计算梯度更新的时候是基于一个batchsize，但是总体参数更新是所有worker的平均结果，相当于基于batchsize*num_worker 大小的梯度更新，更新方向更加接近真实方向，模型收敛平稳。每次梯度更新慢，但是收敛需要的更新次数少。

缺点： 通信开销很大，短板效应

 

**异步参数更新：**

任何一个worker计算完参数更新就把信息发送给ps，ps立即按照信息进行参数更新，以后worker pull到的参数就是更新后的参数。但是这就产生了过期梯度的问题，假设一个worker的计算速度很慢，拿参数的时候拿的v1版本的参数，在计算期间，ps上的参数已经更新到了v3，那么ps就会根据v1版本参数计算得到的梯度来更新v3版本的梯度，得到v4版本，这就会产生震荡，但是最后一般都还是收敛。过期梯度问题可以设置阈值，如果worker上参数的版本比ps的版本差距若干版本以上，则放弃该次更新，防止某一个worker运行速度慢导致梯度更新收过期梯度影响过大。

优点：异步更新没有短板效应，更新速度快

缺点： 更新速度快不意味着收敛速度快，过期梯度问题





## Others

1: **查看tensor的值**，直接键入print(tensorName) 是不行的，这样只会返回tensorname， shape，dtype等信息，不会输出具体数值

```
tf.Print( 
input_, 
data, 
message=None, 
first_n=None, 
summarize=None, 
name=None 
)

input: 通过节点的一个tensor
data: 需要输出的tensorList
message: 消息前缀
first_n: 只记录first_n
summarize: 对每个tensor打印的数量
name:op的名字

return 返回一个input相同的tensor
```

2: 得到tensor的值

```
sess = tf.Session()
res = sess.run(
  fetches,
  feed_dict= None,
  options = None,
  run_metadata = None
)

在使用中尽量少的使用sess.run() 尽量使用一次run 返回多个值

fetches: 一个需要执行的op列表
feed_dict: 一个key-value结构，为之前占位的tensor临时赋值，占位使用tf.placeholder()
通过feed_dict的赋值方式可以实现按照某种特定的计算形式进行不同输入的计算工作

还有一种得到tensor值的方式 tensor.eval()
t.eval() is a shortcut for calling tf.get_default_session().run(t)
```

 

## DEBUG

Bug: TypeError: Can not convert a float32 into a Tensor or Operation

```
user_batch, item_batch, label = create_pipeline(input_file,  num_epochs=10)
with tf.Session() as sess:
    sess.run([init_op, local_init_op])

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    try:
        #while not coord.should_stop():
        while True:
            user, item, label = sess.run([user_batch, item_batch, label_batch])
            print(user, item, label)
            print('--------------')
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop() 
# 然后就会报 以上错误
# 因为一个很蠢的原因， create_pipeline 返回值中的label 和sess.run（label）的输出值同名了，也就是把sess.run
# 之后的结果赋给了一个tensor，就会报错
```

 