---
layout: post
title: Spark&Scala
category: Resource
tags: spark, scala
description: 
---



# Spark

Hadoop 和 Spark的关系：

hadoop的大致框架：由HDFS负责静态数据的存储，通过MapReduce将计算逻辑分发到数据节点进行数据计算和价值发现。

Spark用于大数据量下的迭代式计算，Spark的出现是为了配合Hadoop而不是取代Hadoop；Spark比Hadoop更快的原因是Hadoop把计算中间结果从内存写入硬盘，然后下次迭代之前从硬盘读入, Spark全程的计算数据都存在内存，内存不足时溢出到硬盘中，直到计算出最终结果，然后把结果写入磁盘。

**Spark系统架构**

用户发起一个application，经过master节点，master节点上常驻master守护进程和driver进程，driver负责执行application中的main函数，并且创建SparkContext；master节点负责将串行任务变成可并行执行的任务集tasks，同时负责处理error。master节点将tasks分发到不同的worker nodes，worker nodes 存在一个或者多个executor进程，每个executor进程还有一个线程池，每个线程负责一个task，根据worker node的CPU 核数，可以最多并行等于CPU核数的task。

![img](http://www.jiangwq.com/wp-content/uploads/2019/01/spark-300x153.jpg)

**RDD（Resilent Distributed Datasets）**

spark API 的所有操作都是基于RDD的；数据不止存储在一台机器上，而是分布在多台机器上。RDD是一种“只读”的数据块，任何对RDD的操作都会产生一个新的RDD；RDD之间的transformation和action都会被记录成lineage，lineage形成一个有向无环图（DAG），计算过程不需要将中间结果放入磁盘保证容错，如果某个节点的数据丢失，按照DAG关系重新计算即可。

![img](http://www.jiangwq.com/wp-content/uploads/2019/01/855959-20160920113159606-662705486-280x300.png)

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

**SPARK缓存机制**

每次对一个RDD进行操作的时候，都是按照lineage从头开始计算的，这一点和TF有点像，为了得到一个op的结果，每次sess.run(op)都是从头计算。那么如果一个RDD需要被经常使用，就需要使用缓存机制rdd.cache()默认是内存中缓存，cache内部调用默认的persist操作。cache的RDD 会一直占用内存，需要使用unpersist释放掉

![img](http://www.jiangwq.com/wp-content/uploads/2019/01/v2-76d18f5711274ad4b5e33f0e8741d000_r-300x219.jpg)

**Spark 作业调度：**

spark目前的资源分配有三种：

1 standalon 原生资源管理，由master负责资源分配

2 apache mesos： 和hadoop mr兼容性好的一种资源调度框架

3 hadoop yarn： 指yarn中的额resource manager

yarn 是一个资源调度平台，负责为运算程序提供服务器运行资源，相当于个分布式的操作系统，yarn并不知道用户提交的程序逻辑，只负责提供资源的调度；其中resource manager相当于master节点，具体提供运算资源的事node manager，yarn和用户程序完全解耦，可以运行各种类型的分布式运算程序，如Mapreduce，spark。

spark默认采取FIFO的调度策略。用一个queue保存已经提交的jobs，如果top的job占用所有资源的话，后面的jobs不管多么小都要等待资源释放

后续spark版本支持公平策略调度，采用round robin方式为每个job分配执行的tasks。sparks支持将不同的jobs划分到不同的调度池中，可以为每个调度池设置不同的属性调度池共享集群的资源，每个调度池内部，job是默认FIFO的方式运行的。

 

# Syntax

- val 用于定义常量 val x : int = 5 等价与 val x = 5，var用于定义变量, **值得注意的是使用val定的变量如果在后续程序中不小心进行了修改操作，那么这些操作会丢失，原常量的值不变**。
- scala 的数据类型的关键词首字母大写如 Int, Short...
- **anonymous function** “=>” 表示，符号左边代表参数列表，符号右边表示函数体
- **零参方法和无参方法**： 在定义函数时，例如定义def test() {...} 这是零参方法，在调用时还是需要加上括号 value = test(), 如果想省略括号就需要在定义方法的时候显式的表明方法是无参方法。 test = {...}, 之后调用test方法时候不需要加上括号
- **方法覆盖的时候必须显式的写上override修饰符**，避免accidental overriding
-  **string interpolation**： method1: "lambda equals to" + lambda + "."; method2: s"lambda equals to $（lambda）." “s” 开头，$表示后面的refer to external data.
- **() is the fine representation of not having a value**. e.g. val x = ()
- **Tuple**: definition: ordered container of two or more values,there is not way to iterate through and change element in a tuple. elements in a tuple may have different data types. **访问tuple element 通过 t._index， index 从1 开始**
- 定义或者修改val，var的时候可以用multiple expression e.g. val test = { val x = 6; x +10}
- if-else 用于赋值 val x = if(a>b) a else b
- **match 表达式** val max = x>y match{case true =>x ; case false => y}
- pattern alternative 可以共享case block。 e.g. val day = days match{case "MON"|"TUE"|"WED"|"THU"|"FRI" => "weekday" ; case "SAT"|"SUN" => "weekend"; case other => "other"} ， other 可以用 _ 替代
- iterator 通过to， until 关键词创建 **"to"创建一个inclusive list "until" 创建一个exclusive list**， by关键词指定间隔
- **value binding** 有的时候一些for循环内的变量在每次循环是的都要更新，scala提供一种新的方式在完成更新，与其放在函数内 不如放在for循环语句内 e.g. for( x <- 1 to 10; pow = x^2){ println(pow)}
-  **function**: def functionName(v1: type1 , v2: type2 ..): returnType = { function body}
- **procedure:** 没有返回值的function 就是procedure，如果函数只有一行，可以在定义时省去returntype，如果有多行，必须加上Unit 作为returnType, 或者省去Unit 和 = 号
- **vararg :** parameters that can match zero or more arguments from the caller.
- **type parameter:** 为了增加函数的复用性，引入类型参数；def functionName[A,B,C](a:A, b:B) : C = {...} ; A,B,C是type paremeter，这样input的类型和函数的返回值类型都是可变的
- **function type** : 函数的类型就是输入类型到输出类型 e.g. String => String; 主要是用来把函数当作“值” 赋给val 或var；e.g. val test : String=> Int = func
- **function literal (lambda expression) :** (v1: type1, v2:type2...) => {expression}
- 和c++ ， java一样 整形之间的除法是取整的，scala的类型转换不是(double) [这是java的写法] 而是  intNum.toDouble. 
- **to,until** : for(i< 1 to 10){println} 会打印1到10（包含10）for(i<-1 until 10){} 打印1 到9 

# Immutable Collection

### List

- **Immutable single linked list**
- val test = List(v1,v2,v3,v4...) elements 的type可以不同， list元素不可变
-  **Indexing**: L(index); L.head 返回首个元素， **L.tail 返回剩下的元素**
- **Nil** is singleton instance of List[Nothing], 可用于判断List 是否为空，和List.isEmpty 等价
- **List 1 ：：：List 2** 合并， 但是会去掉重复元素
- **List 1 ++ List2** , append操作 不会去掉重复元素
- **List:+ value** append操作， 由于list 是immutable的，所以不能直接用list + value 实现append， 必须需要“：”符号
- distinct 去重， filter 过滤， partition 按照规则把list分为两个tuple， reverse 反转；sortBy 按照规则排序 List.sortBy(_.size)

### Set

- **immutable, unordered , a collection of unique elements**

### Map

- val M: Map[keytype1 , valuetype] = Map()
- 使用“+ （key->value）" 来添加键值对，“- key” 来去掉键值对

# Mutable Collection

> 需要Import 下列package
>
> collection.mutable.Buffer
>
> collection.mutable.Set
>
> collection.mutable.Map
>
> val test = collection.mutable.Buffer(); test += value 就可以实现append操作

# Collection Function

- map

  : 对RDD集合中的每个元素应用指定的function，一般来说，如果想实现一个for循环对一个iterable结构进行遍历执行某个操作，都可以用map代替。执行结果替代元素值, 

  值得注意一点就是原List 如果是不可变的类型的话，经过map function是不会改变原来的值的，如果需要保存结果就需要把结果赋值给其他变量

  。

  ```
  val testList = List(1,2,3)
  testList.map(num => num*2)
  // => 符号表示映射
  out=List(2,4,6)
  ```

- foreach

  : 主要用来遍历集合元素输出

  ```
  val testList = List(1,2,3,4)
  testList.foreach(num=>println(num))
  1
  2
  3
  4
  ```

- **collectAsMap**: 把[K,V]类型的RDD转换为Map格式，注意，**如果该RDD太大，会出现Java heap memory超的情况**

- flatten: 

  对象是集合的集合， 把2层嵌套结构展平，超过两层就需要多调用几次，但是不如flatMap常用， stack overflow上说比flatMap more efficient （?）

  ```
  val testList = List(List(1, 2), List(3, 4), List(5, 6))  
  testList.flatten(num => num.tail)
  out: List(2,4,6)
  ```

- flatMap

  : 和flatten差不多

  ```
  val testList = List(List(1, 2), List(3, 4), List(5, 6))  
  testList.flatten(num => num.map(num=>num*2))
  out: List(2,4,6,8,10,12)
  ```

- Join, leftOuterJoin, rightOuterJoin

  ```
  #基本语法
  RDD.join(another RDD)
  只有Key , Value形式的RDD可以进行join, 也就是二元形式.如果一个RDD形如 (V1, V2, V3, V4)是不能join的
  可以转换为（V1,(V2,V3,V4)）的形式
  如果两个RDD形如 (v1, (v2,v3,v4)),  (v1,(v5,v6)) 两个RDD依据v1进行join操作之后的结果 R1.join(R2)
  (v1,((v2,v3,v4),(v5,v6)))
  ```

   

- reduce/reduceByKey

  ```
  //reduce把RDD 的两两元素传递给操作函数，返回一个和元素同类型的值，然后和下一个element进行同样的操作，直到最后一个值。
  //例子：
  //求和 "_"是占位符表示一个element 使用"_"而不是其他符号只是为了简便  此时元素必须为一元，不能为tuple或者其他集合形式
  .reduce(_+_)  
  // 求最大值, 可以把a看做reduce过程中一直在维护的一个变量，这个变量保存当前的最大值
  .reduce( (a,b)=> if(a>b) a else b ) 
  // 如果RDD 中每一个元素是tuple形式 例如两个RDD （1,2,3,4） （5,6,7,8） 想对应位置相加
  .reduce((a,b) => (a._1+b._1, a._2+b._2, a._3+b._3))
  
  //reduceByKey 对象是key-value类型的RDD，返回值也是RDD类型，如果是3元及以上的RDD，需要转换为二元key-value 例如（1,2,3,4）
  //不能直接reduceByKey，先转换为（1，（2,3,4））,"_"占位符代表是value元素
  .reduceByKey(_+_)
  ```

   

## Rules

- object 定义的class 下如果包含main函数，则该object为singleton object，只有一个instance，且instance的名字和class的名字一样
- scala不存在静态成员（no matter methods or fields）scala使用Singleton中的成员来代替静态成员
- scala文件经过scalac编译之后生成的是标准的java class文件
- scala中everything is object，it is a pure object-oriented programming language. 数字是对象，也有自己的方法，1+2 实际上是（1）.+(2) 整数1调用他的“+”方法，传入的参数是2
- scala词法分析器采用long-match算法，最长匹配
- 函数的返回值可以implicitly defined， 根据函数体最后一行代码的类型自动判断
- 和动态语言不同，scala中if（0），if("") 会报错mismatch， 并不能automaticlly converted into boolean type.
- 尽管var 定义的变量值可以reassign at anytime。 但是var的类型是不能变的
- statement vs expression expression 有返回值， statement 没有返回值unit类型

##  

## Option

option类通常作为scala集合类型（List，Map）的返回类型，option类型有两个子类，None和Some。当函数成功返回一个string的时候，回传some（string），如果程序返回为None，则没有得到想要的字符串，例如Map.get(key) key不在map中的时候返回None. 集合.get方法返回值不是具体的数值，或字符串，而是一个option结构，通过option.get方法获得具体的值.

 为什么要加option这个结构，把具体的返回值封装为一个结构，为了得到具体的值还需要多个get方法？ 大概是出于提醒人们处理NPE异常的目的，并且option结构后常常接flatMap，如果option为空，不执行flatMap，否则执行，省去一个判空的步骤。网上说的option结构解决NPE问题，实际不准确。如果实在想得到map中一个key对应的value值，

```
获得一个map中key对应的value具体值
Map.get(key).get
如果确定key对应的value不为null的话 可以直接写成
Map(key)
```

 

## **Spark中的DataFrame**

spark 中的dataframe 和RDD一样也是一个分布式的存储结构，并不是pandas中dataframe in memory 的数据结构

详细对比 http://www.lining0806.com/spark%e4%b8%8epandas%e4%b8%addataframe%e5%af%b9%e6%af%94/

```
# pandas dataframe to spark dataframe
SQLContext.createDataFrame(pandas_df) 

# spark dataframe to pandas dataframe 需要保证spark_df 很小，因为pandas_df 不是分布式的结构，需要全部加载进内存的
pandas_df = spark_df.toPandas() 

# spark.dataframe 虽然是分布式存储的，但是可以显示的指明加载到内存,虽然全部加载到内存，但是类型还是spark.dataframe
# SQLContext.sql('''xxx''')的返回值就是spark.dataframe类型
spark_df.persist() / spark_df.cache()  
```

debug:

在pandas.dataframe 转成spark.dataframe 的时候可能会有‘Can not merge type <xxxxx>’

解决方法： df中存在空值，需要先处理空值，处理完可能还是不行，这个时候就需要强制类型转换，强制保证一个字段下数据的类型一致

## Others

- .conf 文件是文本配置文件，不同的公司有不同的写入格式，一般为key-value 形式

- ```
  FEEDS {
    feedsImpressionWithKeysOfflinePath = "path"
    articleInfoPath = "path"
    id = 10
  }
  ```

  上述代码为.conf文件示例代码， 值得注意一点是 **字段前不要加 val 关键字**