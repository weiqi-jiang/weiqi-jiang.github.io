---
layout: post
title: Spark
category: Resource
tags: spark
description: 
---

## Spark

最有用，最全面的了解方式依然是[中文文档](https://spark-reference-doc-cn.readthedocs.io/zh_CN/latest/programming-guide/sql-guide.html)

### 什么是spark

Hadoop 和 Spark的关系：hadoop的大致框架是由HDFS负责静态数据的存储，通过MapReduce将计算逻辑分发到数据节点进行数据计算和价值发现，Spark用于大数据量下的迭代式计算，Spark的出现是为了配合Hadoop而不是取代Hadoop；Spark比Hadoop更快的原因是Hadoop把计算中间结果从内存写入硬盘，然后下次迭代之前从硬盘读入, Spark全程的计算数据都存在内存，内存不足时溢出到硬盘中，直到计算出最终结果，然后把结果写入磁盘。

### Spark系统架构

用户发起一个application，经过master节点，master节点上常驻master守护进程和driver进程，driver负责执行application中的main函数，并且创建SparkContext；master节点负责将串行任务变成可并行执行的任务集tasks，同时负责处理error。master节点将tasks分发到不同的worker nodes，worker nodes 存在一个或者多个executor进程，每个executor进程还有一个线程池，每个线程负责一个task，根据worker node的CPU 核数，可以最多并行等于CPU核数的task。

![img](/assets/img/resource/spark/spark-arch.jpg)

### Spark缓存机制

每次对一个RDD进行操作的时候，都是按照lineage从头开始计算的，这一点和TF有点像，为了得到一个op的结果，每次sess.run(op)都是从头计算。那么如果一个RDD需要被经常使用，就需要使用缓存机制rdd.cache()默认是内存中缓存，cache内部调用默认的persist操作。cache的RDD 会一直占用内存，需要使用unpersist释放掉

![img](/assets/img/resource/spark/spark-buffer.jpg)

### Spark 作业调度

spark目前的资源分配有三种：

1 standalon 原生资源管理，由master负责资源分配

2 apache mesos： 和hadoop mr兼容性好的一种资源调度框架

3 hadoop yarn： 指yarn中的额resource manager

yarn 是一个资源调度平台，负责为运算程序提供服务器运行资源，相当于个分布式的操作系统，yarn并不知道用户提交的程序逻辑，只负责提供资源的调度；其中resource manager相当于master节点，具体提供运算资源的事node manager，yarn和用户程序完全解耦，可以运行各种类型的分布式运算程序，如Mapreduce，spark。

spark默认采取FIFO的调度策略。用一个queue保存已经提交的jobs，如果top的job占用所有资源的话，后面的jobs不管多么小都要等待资源释放

后续spark版本支持公平策略调度，采用round robin方式为每个job分配执行的tasks。sparks支持将不同的jobs划分到不同的调度池中，可以为每个调度池设置不同的属性调度池共享集群的资源，每个调度池内部，job是默认FIFO的方式运行的。

### RDD

spark API 的所有操作都是基于RDD的；数据不止存储在一台机器上，而是分布在多台机器上。RDD是一种“只读”的数据块，任何对RDD的操作都会产生一个新的RDD；RDD之间的transformation和action都会被记录成lineage，lineage形成一个有向无环图（DAG），计算过程不需要将中间结果放入磁盘保证容错，如果某个节点的数据丢失，按照DAG关系重新计算即可。

![img](/assets/img/resource/spark/rdd-backup.png)

如图所示：RDD1的partition2 是根据RDD0的partition2计算而来，如果RDD1的partition2丢失，其他partition不需要重新计算，只需要从RDD0的partition2 重新计算一遍即可。

**RDD的操作函数（operation）**

Tranformation： Map,filter, groupBy, join, union,reduce,sort,partitionBy 返回值还是RDD，不会马上提交spark集群运行，只是记录这些操作，等到有action操作的时候才会真正启动计算过程。

Action： count,collection,take,save,show 返回值不是RDD，会形成DAG，立即提交Spark集群运行并返回结果，形成DAG的先决条件是最后一个操作是Action操作

**Shuffle and stage**

RDD在lineage有两种依赖方式：

> - 窄依赖是指父RDD的每一个分区最多被一个子RDD的分区所用，表现为一个父RDD的分区对应于一个子RDD的分区 
>   或多个父RDD的分区对应于一个子RDD的分区，也就是说一个父RDD的一个分区不可能对应一个子RDD的多个分区。 
>   1个父RDD分区对应1个子RDD分区，这其中又分两种情况：1个子RDD分区对应1个父RDD分区（如map、filter等算子），1个子RDD分区对应N个父RDD分区（如co-paritioned（协同划分）过的Join）。
> - 宽依赖是指子RDD的分区依赖于父RDD的多个分区或所有分区，即存在一个父RDD的一个分区对应一个子RDD的多个分区。 
>   1个父RDD分区对应多个子RDD分区，这其中又分两种情况：1个父RDD对应所有子RDD分区（未经协同划分的Join）或者1个父RDD对应非全部的多个RDD分区（如groupByKey）。 

在容错机制中，如果一个节点死机了，而且运算窄依赖，则只要把丢失的父RDD分区重算即可，不依赖于其他节点。而宽依赖需要父RDD的所有分区都存在，重算就很昂贵了。可以这样理解开销的经济与否：在窄依赖中，在子RDD的分区丢失、重算父RDD分区时，父RDD相应分区的所有数据都是子RDD分区的数据，并不存在冗余计算。在宽依赖情况下，丢失一个子RDD分区重算的每个父RDD的每个分区的所有数据并不是都给丢失的子RDD分区用的，会有一部分数据相当于对应的是未丢失的子RDD分区中需要的数据，这样就会产生冗余计算开销，这也是宽依赖开销更大的原因。因此如果使用Checkpoint算子来做检查点，不仅要考虑Lineage是否足够长，也要考虑是否有宽依赖，对宽依赖加Checkpoint是最物有所值的。

driver根据是否有shuffle（类似reduceByKey，join）操作将作业分为不同的stage，stage的边缘就是shuffle操作发生的地方。每个stage执行一部分代码片段，并为每个stage创建一批task，这些task分配到各个executor进程中执行，每个task执行同样的处理逻辑，只是处理数据块不同的部分罢了。一个stage的所有task执行完之后，将中间结果写入到磁盘，下个stage的输入就是上一个stage的输出，此处有大量的IO消耗，所以应该尽量减少shuffle操作。

### Parquet

parquet是一种**面向分析**的，通用的**列式存储格式**，首先就要区别列式存储和行式存储的区别，顾名思义列存储就是以列为序列存储数据，行存储以遍历行为序列存储数据，总结起来列存储更适合**OLAP**，行存储更适合**OLTP**. OLAP(On-Line Analytical Processing) 支持复杂的分析操作，偏向于决策支持；OLTP(On-Line Transaction Processing)是传统的关系型数据库的主要应用，偏向基本的，日常的事务处理。简单的来理解的话，OLTP的常用场景是录入，删除，修改，查找一条完整的记录，OLAP是提取聚合后的数据进行分析操作。所以很明显，在spark大数据分析场景下，更多的是使用列存储格式。

parquet或者说列式存储的特点主要体现在

- 列裁剪，只加载需要的列，减少IO操作
- 谓词下推，将过滤表达式尽可能的下沉到靠近数据源的地方，减少map,reduce操作的数据量
- 优化存储空间，同一列的数据类型相同，可以使用更合适的编码格式，降低存储空间

相关操作

```scala
// sparkSession 表示SparkSession对象

//读取
val df = sparkSession.read.parquet("/path/to/parquetfile.parquet") // df 为dataframe类型

/*
指定写入格式 Append,ErrorExists(default)存在报错， Ignore 文件存在则什么都不做 Overwrite
写入后parquetfile.parquet不是文件而是文件夹，包含四个以上文件，part个数取决于指定partition的个数
_common_metadata  
_metadata  
part-r-00000-ad565c11-d91b-4de7-865b-ea17f8e91247.gz.parquet  
_SUCCESS
可检查_SUCCESS文件是否生成来觉得是否调度后续任务，加载时，只需要给出文件夹的路径，不需要具体文件路径
*/
df.write.parquet("/path/to/parquetfile.parquet")
df.write.mode(SaveMode.Append).parquet("/path/to/parquetfile.parquet")
```

**Reference**<br>[Apache Parquet 干货分享](https://cloud.tencent.com/developer/article/1498575)<br>[传统的行存储和（HBase）列存储的区别](https://blog.csdn.net/youzhouliu/article/details/67632882)<br>[为什么列存储数据库读取速度会比传统的行数据库快？](https://www.zhihu.com/question/29380943)<br>[OLAP、OLTP的介绍和比较](https://blog.csdn.net/zhangzheng0413/article/details/8271322)<br>[谓词下推](https://blog.csdn.net/baichoufei90/article/details/85264100)<br>[Spark：DataFrame保存为parquet文件和永久表](https://blog.csdn.net/xuejianbest/article/details/85775442)<br>[Spark入门：读写Parquet(DataFrame)](http://dblab.xmu.edu.cn/blog/1091-2/)

