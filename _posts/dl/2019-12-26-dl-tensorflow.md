---
layout: post
title: TF(TensorFlow) Boys
category: deep learning
tags: tensorflow
description: TensorFlow 用法
---



# 理论

深度模型强调两个点： ***多层，非线性***

非线性：多层线性模型等价于一层线性模型，线性模型解决的问题很有限，所有强调非线性

多层： 这里的多层并不是指很多个hidden layer，实际上一层hidden layer就足够了，如果没有hidden layer 就是perceptron, perceptron是不能解决异或问题的

#  

# 入门

### 计算图

> tensorflow每一个计算都是都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系

系统自动维护一个默认的计算图,tf.get_default_graph 得到这个图。通过tf.Graph() 创建新的图。不同图上的张量和运算都不共享

### 张量

> 张量在功能上相当于多维数组，但是实现并不是直接采用多维数组的形式，只是对tensorflow中运算结果的引用，并没有保存真正的数字

通过调用tf.Tensor.eval(tensor,session = sess) 可以得到张量的具体值,或者tensor.eval(session = sess)。张量主要保存三个属性： **名字，维度和类型**

### 会话

**tensorflow会自动生成默认的计算图，但是不会生成默认的session**，如果没有指定模型session，with sess.as_default(), tf.Tensor.eval()需要指定session才能得到tensor的值

config = tf.ConfigProto() 来配合生成的会话，最常用allow_soft_placement=True,这个参数允许GPU在特定情况下可以在CPU上运行，而不是报错。

**tf中collection的概念**

```
tf.GraphKeys.VARIABLES   所有变量
通过tf.global_varibles() 获取
tf.GraphKeys.TRAINABLE_VARIABLES  可学习的变量
通过tf.trainable_variables() 获得
tf.GraphKeys.SUMMARIES  日志生成相关的张量
tf.GraphKeys.QUEUE_RUNNERS 处理输入的QueueRunner
tf.GraphKeys.MOVING_AVERAGE_VARIABLES  所有计算了滑动平均的变量
```

**TF随机数生成函数**

```
tf.random_normal   正太分布；   参数：mean,stddev,dtype,seed,name
tf.truncated_normal 受限正太分布，如果偏移均值两个标准差，重新随机；
参数： mean,stddev,dtype,seed,name
tf.random_uniform   均匀分布； 参数： maxval，minval,dtype,seed,name
tf.random_gamma   gamma分布  参数，alpha,beta,dtype
```

 

**TF中变量的创建**

```
# 主要是两个方式 tf.Variable tf.get_variable

# 用随机数生成器来赋值
v1 = tf.Variable(tf.truncated_normal([d1,d2], stddev=0.1))

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

 

**TF 持久化**

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

 

## Protocol Buffer

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

 

# TF基本示例

```
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

 

# IO stream

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

```
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

 

 

# TFRecord

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

![img](http://www.jiangwq.com/wp-content/uploads/2019/01/20180915164530504-300x193.png)

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

 

##  

# SparseTensor Class

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

 

 

 

##  

# Variable

```
tf.Variable(
 initial_value = None , trainable = True, ....
)

tf.get_variable（
  name, shape = None, dtype = None, initializer = None ,
  regularizer = None,trainable = True ...
）

# get_variable 和Variable的区别，Variable 如果命名冲突，系统自动处理，get_varibale 报错
```

 

所有的变量会被自动加入到GraphKeys.VARIABLES这个collection中，通过tf.global_variables()这个函数可以得到当前graph上所有的变量。如果在变量声明函数中trainable参数指定为True, 那么这个变量会被加入到GraphKeys.TRAINABLE_VARIANCES, 并通过tf.trainable_variables()得到所有需要优化的参数。TensorFlow 中的优化算法会默认优化GraphKeys.TRAINABLE_VARIABLES 集合中的参数。

# Train.

- string_input_producer(): output strings to

   filename queue

   for an input pipeline

  ```
  tf.train.string_input_producer(
      string_tensor,
      num_epochs=None,
      shuffle=True,  // shuffle by default
      seed=None,
      capacity=32,
      shared_name=None,
      name=None,
      cancel_op=None
  )
  ```

- Coordinator 线程协调器，用来管理session中的multiple thread

- start_queue_runners, 启动tensor的入队线程，将tensors 推入filename queue ，只有调用该函数之后，tensor才真正被推入内存序列中，否则由于内存序列为空，数据流图会一直处于等待状态。

  ```
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runner(sess,coord)
  
  #
  #do something
  #
  
  coord.request_stop()# 发出终止所有进程的信号
  coord.join(threads) #把开启的进程加入主进程，等待threads结束
  ```

- tf.train.MonitoredTrainingSession()

  ```
  MonitoredTrainingSession(
      master='',
      is_chief=True,
      checkpoint_dir=None,
      scaffold=None,
      hooks=None,
      chief_only_hooks=None,
      save_checkpoint_secs=600,
      save_summaries_steps=USE_DEFAULT,
      save_summaries_secs=USE_DEFAULT,
      config=None,
      stop_grace_period_secs=120,
      log_step_count_steps=100
  )
  
  Args:
   is_chief：用于分布式系统中，用于判断该系统是否是chief，
   如果为True，它将负责初始化并恢复底层TensorFlow会话。如果为False，它将等待chief初始化或恢复TensorFlow会话。
  
   checkpoint_dir：一个字符串。指定一个用于恢复变量的checkpoint文件路径。
  
   scaffold：用于收集或建立支持性操作的脚手架。如果未指定，则会创建默认一个默认的scaffold。它用于完成图表
  
   hooks：SessionRunHook对象的可选列表。可自己定义SessionRunHook对象，也可用已经预定义好的SessionRunHook对象，
   如：tf.train.StopAtStepHook()设置停止训练的条件；tf.train.NanTensorHook(loss):如果loss的值为Nan则停止训练；
   chief_only_hooks：SessionRunHook对象列表。如果is_chief== True，则激活这些挂钩，否则忽略。
  
   save_checkpoint_secs：用默认的checkpoint saver保存checkpoint的频率（以秒为单位）。
   save_checkpoint_steps: 如果save_checkpoint_step 和 save_checkpoint_secs 都设为none 则不会调用 默认的 saver去保存cp。
   这个很重要 如果不同时指定为none，如果有其他程序加载最新的cp，进行一些操作之后，会产生一个空间占用较小的新的cp污染原来的目录
  
   save_summaries_steps：使用默认summaries saver将摘要写入磁盘的频率（以全局步数表示）。
   如果save_summaries_steps和save_summaries_secs都设置为None，则不使用默认的summaries saver保存summaries。默认为100
   
   save_summaries_secs：使用默认summaries saver将摘要写入磁盘的频率（以秒为单位）。
   如果save_summaries_steps和save_summaries_secs都设置为None，则不使用默认的摘要保存。默认未启用。
  
   config：用于配置会话的tf.ConfigProtoproto的实例。它是tf.Session的构造函数的config参数。
  
   stop_grace_period_secs：调用close（）后线程停止的秒数。
  
   log_step_count_steps：记录全局步/秒的全局步数的频率
  
  Returns:          
         一个MonitoredSession（） 实例。
  ```

- placeholder

#  

# Distributed TF

单机多GPU的工作模式：

由CPU 负责把batch发给多GPU，多个GPU 负责计算梯度更新，等待所有GPU 运算完毕，梯度更新数据发送给CPU，CPU 计算平均更新梯度，进行梯度更新，接着发送新的batch给多个GPU，时间消耗取决于最慢的GPU 和CPU,GPU 通信时间。

 

多机多GPU 模式：

当数据量急剧增大的时候，参数更新的速度就成了一个大问题，于是单机模式下单CPU 进行参数更新的模式就行不通了，引入Parameter Server (PS)的概念。组成集群进行梯度更新

Tensorflow 分布式给予gRPC通信框架(google Remote Proceduce Call), 简单理解就是把参数进行打包上传给服务器，服务器端进行运算，运算完之后把运行结果打包传回客户端。

 

### 分布式并行策略：

模型并行: 将模型的不同部分并行分发到不同的worker上训练，但是模型层与层之间是存在串行关系的，所以一般只有在不存在串行的部分进行模型并行

数据并行： worker使用的是同一个graph，但是数据是分块发给不同worker的，分别进行计算，参数更新模式有同步和异步之分。

### 参数更新

worker和ps之间的通信都由worker发出pull和push请求，什么时候发出请求是由scheduler调度。

同步参数更新：

workers 运算得到梯度更新参数，等到所有worker跑完一个batchsize，把参数更新信息发送给ps，计算平均梯度，ps进行梯度更新，然后把更新后的参数值传回worker，循环进行。

优点： 虽然单个worker计算梯度更新的时候是基于一个batchsize，但是总体参数更新是所有worker的平均结果，相当于基于batchsize*num_worker 大小的梯度更新，更新方向更加接近真实方向，模型收敛平稳。每次梯度更新慢，但是收敛需要的更新次数少。

缺点： 通信开销很大，短板效应

 

异步参数更新：

任何一个worker计算完参数更新就把信息发送给ps，ps立即按照信息进行参数更新，以后worker pull到的参数就是更新后的参数。但是这就产生了过期梯度的问题，假设一个worker的计算速度很慢，拿参数的时候拿的v1版本的参数，在计算期间，ps上的参数已经更新到了v3，那么ps就会根据v1版本参数计算得到的梯度来更新v3版本的梯度，得到v4版本，这就会产生震荡，但是最后一般都还是收敛。过期梯度问题可以设置阈值，如果worker上参数的版本比ps的版本差距若干版本以上，则放弃该次更新，防止某一个worker运行速度慢导致梯度更新收过期梯度影响过大。

优点：异步更新没有短板效应，更新速度快

缺点： 更新速度快不意味着收敛速度快，过期梯度问题

# Others

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

 

# **DEBUG**

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

 