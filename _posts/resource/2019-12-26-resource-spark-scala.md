---
layout: post
title: Spark&Scala
category: Resource
tags: spark, scala
description: 
---



## Spark

### 什么是spark

Hadoop 和 Spark的关系：hadoop的大致框架是由HDFS负责静态数据的存储，通过MapReduce将计算逻辑分发到数据节点进行数据计算和价值发现，Spark用于大数据量下的迭代式计算，Spark的出现是为了配合Hadoop而不是取代Hadoop；Spark比Hadoop更快的原因是Hadoop把计算中间结果从内存写入硬盘，然后下次迭代之前从硬盘读入, Spark全程的计算数据都存在内存，内存不足时溢出到硬盘中，直到计算出最终结果，然后把结果写入磁盘。

### Spark系统架构

用户发起一个application，经过master节点，master节点上常驻master守护进程和driver进程，driver负责执行application中的main函数，并且创建SparkContext；master节点负责将串行任务变成可并行执行的任务集tasks，同时负责处理error。master节点将tasks分发到不同的worker nodes，worker nodes 存在一个或者多个executor进程，每个executor进程还有一个线程池，每个线程负责一个task，根据worker node的CPU 核数，可以最多并行等于CPU核数的task。

![img](/assets/img/resource/spark/spark-arch.jpg)

### RDD（Resilent Distributed Datasets）

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

## Scala

[官方文档](https://docs.scala-lang.org/zh-cn/)

### Base

1. val, var 初始化

```scala
//al，var都必须要定义时赋值，var可以使用默认初始化,但是必须指定数据类型，否则报错
var a:String = _  //初始为null
var a:Int = _ // 初始为0 

```

2. print和println的区别在于print 不会再内容后追加换行符，println会追加换行符
3. 标准输入

```scala
import scala.io
// readLine 可以接受一个参数作为提示字符串
val name = StdIn.readLine("your name is : ")
//当然也可以指定数据类型，不接受提示字符串
val t1 = StdIn.readInt()
val t2 = StdIn.readChar()
val t3 = StdIn.readBoolean()
val t4 = StdIn.readDouble()
val t5 = StdIn.readLong()
```

4. 当val 用**lazy** 修饰时，初始化被推迟直到第一次取值

```scala
lazy val a = 0
```

5. **val 不可变指的是不能改变初始化指向的类的实例对象，但是对象的值可以变；var可变指的是可以改变初始化指向的类的实例对象，而且实例对象自己也可以变**

### Run

```shell
# 编译, 生成xxx.class文件
scalac xxx.scala 
#运行
scala xxx
# 可能会存在不能运行的情况，因为当前路径不在CLASSPATH下，需要显式添加进去，也可以在环境变量中添加当前路径
scala -cp . xxx
```

Reference<br>[Can compile scala programs but cannot run them](https://stackoverflow.com/questions/27998824/can-compile-scala-programs-but-cant-run-them)

### Operator

scala中有一个特殊的点在于**操作符实际上是方法**，a 方法 b  是a.方法(b)的简写形式。一个重要规则是，**如果一个无参方法并不修改对象，调用时不用写扣号**

```scala
val a = 4 
val b = a + 3 // b = 7
val c = a.+(3) // b= 7 使用方法和使用操作符的结果一致 
```

### String

字符串元素访问使用“()” 并不是“[]”, 索引的过程可以看成一个根据index拿到字符的过程，这个过程是通过一个“映射函数”来完成的，所以使用“()”。

```scala
val a = "hello world"
a(3) // "l"
a.apply(3) // 等价于a(3)

```



### Function

**返回值**，普通函数不明显使用return指明返回值，如果有返回值，最后一行就是返回值

```scala
// 显式指明返回值类型
def test(x: Int, y: Int): Int  = {
  x + y
}
// 不指明返回值类型，则没有返回值，相当于返回值类型为Unit， 可以不写 = 
def test1(x: Int, y: Int){
    println(x+y)
}
// 带默认值的函数
def test2(x: Int, y:Int=10): Int = {
    x+y
}

```

**零参方法/无参方法/变长参数方法**

```scala
// 零参方法
def foo1() = {
	println("hello scala")
}
foo1()
// 无参方法，调用的时候不需要加括号
def foo2 = {
    println("hello scala")
}
foo2
// 变长参数方法
def sum(args: Int*) = {
    var result = 0
    for (arg<-args){
        result += arg
    }
    result
}
sum(1,2,3,4,5)
//不能直接传入1 to 5，需要告诉编译器参数被当成序列处理
sum(1 to 5: _*)
```

**匿名函数**，可以简化写函数的过程，类似于lambda 表达式

```scala
// 最简单的形式
(x:Int,y:Int) => x+y
//可以赋值并指明函数数据类型，写出来是为了一眼看出输入和输出的类型
val test:(Int, Int)=>Int = (x:Int, y:Int) =>x+y
// 当然也可以不写数据类型
val test1 = (x:Int, y:Int) => x+y

```

**部分引用**，使用下划线‘—’部分应用一个函数，返回值为另一个函数，例子中定量add2相当于x固定为2的adder函数

```scala
def adder(x:Int,y:Int): Int = {
    x+y
}
val add2 = adder(2, _:Int)
add2(5)
```

**柯里化函数**,  把原来接受两个参数的函数变成接受一个参数的函数的过程，新函数返回值是以原有第二个参数为参数的函数

```scala
def add1(x:Int, y:Int): Int = {
	x+y
}
add1(1, 2)
//柯里化
def add2(x:Int)(y:Int): Int = {
    x+y
}
add2(1)(2)
//相当于
def add3(x:Int, y:Int): Int = {
    (y:Int) => x +y
}
```

**类型参数化(泛型)**

```scala
// 整数加法
def add(x: Int, y: Int): Int = {
    x + y
}
// 字符型加法 
def add(x: String, y: String): String = {
    x + y
}
// 可以看出来上面的写法很麻烦，如果要实现int和string 需要再写定义新的函数，我们可以把数据类型参数化
def add[a](x: a, y: a): a = {
    x+y
}
def add[a,b](x: a, y: b) = {
    x+y
}
```

Reference<br>[scala泛型](https://fangjian0423.github.io/2015/06/07/scala-generic/)

### Control Structure

**if...else...** 

```scala
// 三元表达式
val a = if (b>1) 1 else 0
// 返回空值
val a = if(x>0) 1 else ()
```

**while loop**

```scala
val n = 10
var r = 0
while(n>0) {
    r = r * n
    n -= 1
}
```

**for loop**

```scala
/*
如果循环中出现全局变量相同的变量，局部变量遮挡全局变量
i前面不需要用val var修饰，类型取决于 后面集合/迭代器的类型
*/

for (i <- 1 to 10){
    print(i)
}

// 嵌套for loop 多个生成器用分号隔开
for(i <- 1 to 3; j <- 1 to 4){print(i*10+j)}

// 嵌套for loop 条件过滤
for(i <- 1 to 3; j <- 1 to 4 if i != j){print(i*10+j)}

// 可以添加任意多的变量
for(i <- 1 to 3; from = 4-i; j <- from to 3){print(i*10+j)}

//  称为for comprehension,返回值类型由原始集合相同，c类型Vector，b类型是Array,如果有过滤，加在for括号内
val c = for(i <- 1 to 10) yield i%3 
val a = Array(1,2,3,4)
val b = for(i<-a) yield i*2
val d = for(i<-a if i%2==0) yield i*2

// until 和 to的区别在于until排除最后一个元素, to 是包含最后一个元素的
for(i <- 1 until 10){ print(i)}

// 设置遍历的步长
for(i<-1 to 10 by 2){print(i)}

// 如果遍历一个数组下标可以使用Array.indices
for(i<-b.indices){print(i)}

```

### Class

```scala
// 基本定义
class TestClass(x:Int, y:Int){
	val z: Int = 0
    
    def add(x:Int,y:Int): Int = {
        x + y
    }
}

val c = new TestClass(1, 2)
// 私有字段, private关键词修饰，该类方法可以访问该类的所有对象的私有字段
class Counter{
    private var value = 0
    def lsLess(other: Counter) = value<other.value
}
// 会报错，因为只能访问当前对象的私有字段
class Counter{
    private[this] var value = 0
    def lsLess(other: Counter) = value<other.value
}
```

scala的构造函数分为两个部分，一个是**主构造函数**，一个是**辅助构造函数**。主构造函数是“隐性的”，它是类的方法定义之外的所有代码，也就是“类内能执行的代码”都是主构造函数

```scala
/*
主构造函数主要包含三个部分：
-构造函数参数
-类内被调用的函数
-类内执行的语句和表达式
*/

/* 
主构造函数说明，当new一个testclass实例时，执行了成员变量z的定义，println函数，这些都是主构造函数
主构造函数参数如果没有用val,var修饰，则只是类内不可变参数，不能用class.variable访问
如果用var/val修饰，则是类的成员变量，就和python中的self.variable 一样
*/ 
class TestClass(x:Int, y:Int){
	val z: Int = 0
    
    def add(x:Int,y:Int): Int = {
        x + y
    }
    
    println("hello constructor")
}
val t = new TestClass(1, 2)


/*
辅助构造函数说明，必须以调用先前定义的构造函数或者是主构造函数开始
可以有任意多的辅助构造函数, 实例化时通过传入不同数量或类型的参数实现不同的实例化效果
相当于函数overload
*/
class TestClass(val x:Int, val y:Int){
	val z: Int = 0
    
    def this(x:Int, y:Int, z:Int): Int = {
        this(x, y)
        val z = z
    }   
    def this(x:Int,y:Int,z:Int,t:Int):Int={
        this(x,y,z)
        val t = t
    }
}
val p1 = new TestClass(1,2,3)
val p2 = new TestClass(1,2,3,4)
```

**Singleton pattern** 单例对象只有一个实例，用object 关键词修饰，和惰性变量一样，延迟创建，即第一次被使用时创建,和类的唯一区别在于不能提供构造器参数。如下，test方法在任何地方都可以引用，**创建功能性方法是单例对象的常见用法**, 而且单例对象是**组织静态函数(static function)的有效工具**，单例对象也常常用在工厂模式设计中，详情见后文。

```scala
package pack
object foo {
    def test() = {
        println("hello")
    }
}
// 程序其他地方可以import 单例对象foo方法的test方法
// 体现了面向对象的思想，用一个对象的方法来实现一个通用的函数
import pack.foo.test
test

/* 
单例对象可以在类内定义也可以不在，可以和类具有相同的名称，此时，该对象称为“伴生对象”
类和伴生对象可以相互访问私有特性，必须存在同一个源文件中
*/ 
class Bar(foo: String){
    object Bar {
        def apply(foo:String) = new Bar(foo)
    }
}

object Bar {
    def apply(foo:String) = new Bar(foo)
}


```

**Apply**方法，当遇到$object(p1,p2,...,pn)$这种形式时，apply方法会被调用，通常返回一个伴生对象

```scala
// 调用的是Array这个对象的apply方法Array.apply(10) 返回一个只包含一个元素的Array[Int]
val a = Array(10)
// 调用的是构造器this(10),返回100个null元素
val b = new Array(10)
```

**抽象类** 用**abstract** 关键词修饰，定义了一些方法但是没有实现，且不可被实例化,**抽象类的作用**(个人理解)： 1. 规范化，继承抽象类的子类拥有共同的方法名，不同的开发人员参考同一个规范。2.统一数据类型，比如在简单工厂模式中根据输入实例化不同的类，这些不同的类具有公共的数据类型, 继承抽象类的**子类必须实现抽象类的所有方法**

```scala
abstract class Car {
    def Run():Unit
}
```

**特质(Trait)** 一些共同的字段和行为的组合,和抽象类很像，大多数时候起到相同的功能；**特质和抽象类的区别**在于抽象类可以有构造函数参数，特质没有；抽象类和java代码完全兼容，特质兼容性不佳；特质可以混入不同的类层级，一些与类解耦的常用行为都可以写进特质，比如展示品牌行为，汽车类可以有，冰箱类可以有，卫生纸类也可以有，该行为和类解耦，适合单独提出来写入特质。详情参见Ref2,Ref3.

```scala
abstract class Car {
    def run():Unit
}

trait luxurycar {
    def showoff():Unit
}

class Benz extends Car{
    override def run() = {
        println("run")
    }
}
class Benz2 extends Car with luxurycar {
    override def run() = {
        println("run")
    }
    override def showoff = {
        println("showoff")
    }
}
```

**样本类(Case Classes)**，就是用case 关键词声明的类，**非常适合不可变的数据**，也常用于模式匹配中, 实例化样例类时不需要new关键词，因为有默认的apply方法。样本类的参数是**公开的不可变的**，可以class.variable形式访问. 这其实也体现了“面向对象”的思想，**一些固定的参数组合用一个类去包装**，**可以使用var 但是不推荐**

```scala
case class Bookinfo(id:String)
val t = Bookinfo("123456")
t.id
```

**特殊的apply方法，当一个对象以方法的形式被调用时，scala底层隐式的转换成在该对象上调用apply方法**，因此apply常被称为“注入方法”

```scala
class Foo {}
object FooMaker{
    def apply() = new Foo
}
val a = FooMaker()
//不需要写new 关键词

// # todo
```



**Reference**<br>[scala构造函数](https://www.jianshu.com/p/bb756fd1d2e6)<br>[To trait, or not to trait?](https://www.artima.com/pins1ed/traits.html#12.7)<br>[What is the advantage of using abstract classes instead of traits?](https://stackoverflow.com/questions/1991042/what-is-the-advantage-of-using-abstract-classes-instead-of-traits)

### Data Structure

**Array** 有序，可变，包含重复项,**定长**

```scala
// array声明, 提供初始值时不要用new关键词，使用new关键词修饰时，初始为null
val numbers = Array(1,2,3,4,5)
val numbers = new Array[Int](10)
// array 元素访问
val n3 = numbers(3)
// 修改元素值
numbers(2) = 1
// map 
val biggernum = numbers.map(_ * 2)
// array 合并
val number2 = Array(6,7,8,9)
val numberall = number ++ number2
// count 
numberall.count(_ > 3)
```

**ArrayBuffer** 有序，可变，变长，可包含重复项

```scala
import scala.collection.mutable.ArrayBuffer
val b = ArrayBuffer[Int]()

// append   b = ArrayBuffer(1, 2)
b += 1
b.append(2)

// 一次性添加一个集合,  b = ArrayBuffer(1,2,1,2,3,4)
b.appendAll(Array(1,2,3,4))

// 移除尾部5个元素 b = ArrayBuffer(1,2,1,2)
b.tridEnd(2)

// 指定位置插入, insert(index, value)  b=ArrayBuffer(1,6,2,1,2)
b.insert(1, 6)

//删除， 第二个参数指定删除元素的个数 .remove(index, num) 
b.remove(1)
b.remove(1,2)

//Array 和ArrayBuffer的转换
val c = b.toArray
val d = c.toBuffer
```

**Array Transform**

```scala
val a = Array(1,2,3,4,5,6)
// 满足filter条件的留下来，然后对剩下的元素做map指定的操作，和for，yield功能相同
a.filter(_%2==0).map(2*_)
```

**MultiDim Array**

```scala
val matrix = Array.ofDim[Int](3,4) // 初始为0
val elem = matrix(1)(2)
```

**Map Related**

```scala
// 声明,要保证key-value 对的类型一致
val m1 = Map("k1" -> "v1","k2" -> "v2","k3" -> "v3")
val m2 = Map(("k1","v1"),("k2","v2"),("k3","v3"))
val m3 = collection.mutable.Map[String, String]()
// 访问元素，如果不存在会报错，用contains方法判断,或.getOrElse
val v1 = m1("k1")
val v2 = if(m1.contains("k2")) m1("k2") else 0
val v3 = m1.getOrElse("k3", 0)

// get 返回一个option对象，要么是键对应值，要么是None
val v4 = m2.get("v2")

// 可变映射添加，修改，删除,添加时要保证类型一致
m3("k4") = "v4" 
m3("k3") = "v3-new"
m3.addAll(Array("k5"->"v5","k6"->"v6")))
m -= "k4"

// 修改不可变map，会创建新的不可变映射, 如果是val定义的m1，则需要赋值新val变量，如果是var，则可以覆盖
val new_m1 = m1 + ("k4"->"v4")
val new_m2 = m2 - "k1"
var m1 = Map("k1" -> "v1","k2" -> "v2","k3" -> "v3")
m1 = m1 + ("k4"->"v4")


// 遍历/翻转 键值对, 访问键，值集合
val m = Map("k1"->"v1","k2"->"v2")
for((k,v)<-m){print(v)}
for((k,v)<-m)yield (v,k)
for(v<-m.values)print(v)
for(k<-m.keySet)print(k)

// 排序映射，按插入顺序排序映射
val sm = scala.collection.mutable.SortedMap("1"->"a","2"->"b")
val lm = scala.collection.mutable.LinkedHashMap("1"->"a","2"->"b")
```

**Tuple** 不可变

```scala
/*
在不使用类的情况下把不同类型元素简单组合，和case classes实现的功能类似，只是样本类可以通过名称来获得字段
tuple只能通过下标来访问，且以1为base
*/
val t = ("hello","world")
t._1 // hello

val (word1, word2) = t
val (word1, _) = t

// ZIP 操作, 转成映射
val num = Array(1,2,3)
val char = Array("a","b","c")
val t = num.zip(char)
for((n,c)<-t) print(c*n)
val m = t.toMap
```

**List** 有序，不可变，可包含重复项

```scala
// 声明 
val numbers = List(1,2,3,4,5)
```

**Set** 无序，不可变，不包含重复项

```scala
// 声明
val numbers = Set(1,2,3,4,5)
```



### Pattern Matching

```scala
// 匹配值
val times = "WED"
val time = times match {
    case "MON"|"TUE"|"WED"|"THU"|"FRI" => "WEEKDAY"
    case "SAT"|"SUN" => "WEEKEND"
    case _ => "Other"
}
//或者是
val times = "WED"
val time = times match {
    case i if i == "MON"|"TUE"|"WED"|"THU"|"FRI" => "WEEKDAY"
    case i if i == "SAT"|"SUN" => "WEEKEND"
    case _ => "Other"
}


//匹配类型
def testmatch(o: Any): Any = {
    o match {
        case i: Int => i+1
        case d: Double => d+0.1
        case text: String => text+'s'
    }
}
```

样本类用于模式匹配

```scala
/*
首先定义一个虚基类是为了后面函数定义时有统一的类型
根据输入具体子类的不同，得到不同的结果
*/
abstract class Notification
case class Email(sender: String, title: String, body: String) extends Notification
case class SMS(caller: String, message: String) extends Notification
case class VoiceRecording(contactName: String, link: String) extends Notification

def showNotification(notification: Notification): String = {
  notification match {
    case Email(sender, title, _) =>
      s"You got an email from $sender with title: $title"
    case SMS(number, message) =>
      s"You got an SMS from $number! Message: $message"
    case VoiceRecording(name, link) =>
      s"you received a Voice Recording from $name! Click the link to hear it: $link"
  }
}
val someSms = SMS("12345", "Are you there?")
val someVoiceRecording = VoiceRecording("Tom", "voicerecording.org/id/123")

println(showNotification(someSms))
println(showNotification(someVoiceRecording))  

```

**Reference**<br>[官方文档：模式匹配](https://docs.scala-lang.org/zh-cn/tour/pattern-matching.html)

### Design Pattern

**Factory pattern** 按照对类的抽象程度可以划分为三个类型：**简单工厂模式(Single Factory)，工厂方法模式(Factory Method)，抽象工厂模式(Abstract Factory)**， 简单工厂模式让对象调用者和对象创建过程分离，用工厂类解耦，在工厂类负责逻辑判断，提高可维护性，可扩展性。但是当要修改产品是，需要修改工厂类，违反开闭原则(对于扩展是开放的，对于修改是封闭的)；工厂方法模式，不在工厂类中进行逻辑判断，同时抽象工厂和产品，不同的工厂负责不同的产品，新增产品时，新建并继承抽象产品类，新建并继承抽象工厂类即可，不需要修改现有的类； 抽象工厂模式更加复杂，下文单独说明。

```scala
/*
简单工厂模式，抽象具体产品，在单例工厂类中进行逻辑判断
*/
// 抽象产品角色，所有产品对象的父类
trait Car{
    def brand()
}
// 具体产品角色，工厂所创建的具体实例对象，预先定义宝马类
class BMW extends Car{
    override def brand() = {
        println("BMW")
    }
}
// 预先定义奔驰类
class Benz extends Car{
    override def brand() = {
        println("Benz")
    }
}
// 单例汽车工厂类
object CarFactory {
    def CreateCar(brand: String) =  brand match {
        case "Benz" => new Benz
        case "BMW" => new BMW
    }
}
// 根据不同的要求实例化不同的类
val car1 = CarFactory.CreateCar("BMW")
val car2 = CarFactory.CreateCar("Benz")
car1.brand()
car2.brand()

/*
工厂方法模式, 同时抽象具体产品和工厂，每个产品由对应的工厂负责实例化
*/
trait CarFactory  {
    val createcar
}
object BMWFactory extends CarFactory {
    override val createcar = new BMW
}
object BenzFactory extends CarFactory {
    override val createcar = new Benz
}
BMWFactory.createcar.brand()
BenzFactory.createcar.brand()

```

抽象工厂模式涉及到产品族的概念，假设我们现在有三个不同品牌的三个不同的车型，三个品牌的sport型汽车属于一个产品族，同一个品牌下的三个不同车型属于一个等级结构。抽象工厂一次只消费其中一族产品，同属于一个产品族的产品一起使用。

![absclass](/assets/img/resource/spark/abstractclass.png)

抽象工厂模式的实现流程是，首先建立抽象产品和抽象工厂类，分别实现9个具体产品类，再分别实现三个sportFactory, busFactory, luxuryFactory 三个工厂类

```scala
/*
抽象工厂模式
*/
trait Car {
    def brand(): Unit
}
// sport具体产品类
class BMWSportCar extends Car {
    override def brand() = {
        println("BMW Sport")
    }
}
class BenzSportCar extends Car {
    override def brand() = {
        println("Benz Sport")
    }
}
class AudiSportCar extends Car {
    override def brand() = {
        println("Audi Sport")
    }
}
// bus具体产品类
class BMWBusCar extends Car {
    override def brand = {
        println("BMW Bus")
    }
}
class BenzBusCar extends Car {
    override def brand = {
        println("Benz Bus")
    }
}
class AudiBusCar extends Car {
    override def brand = {
        println("Audi Bus")
    }
}
// luxury 具体产品类
class BMWLuxuryCar extends Car {
    override def brand = {
        println("BMW Luxury")
    }
}
class BenzLuxuryCar extends Car {
    override def brand = {
        println("Benz Luxury")
    }
}
class AudiLuxuryCar extends Car {
    override def brand = {
        println("Audi Luxury")
    }
}
//抽象工厂类
trait carFactory {
    def createcar(): Array[Car]
}
// 具体工厂类
object SportFactory extends carFactory {
    override def createcar:Array[Car] = {
        val bmw = new BMWSportCar
        val audi = new AudiSportCar
        val benz = new BenzSportCar
        val result = Array(bmw,audi,benz)
        result
    }
}
object LuxuryFactory extends carFactory {
    override def createcar: Array[Car] = {
        val bmw = new BMWLuxuryCar
        val audi = new AudiLuxuryCar
        val benz = new BenzLuxuryCar
        val result = Array(bmw,audi,benz)
        result
    }
}
object BusFactory extends carFactory {
    override def createcar: Array[Car] = {
        val bmw = new BMWBusCar
        val audi = new AudiBusCar
        val benz = new BenzBusCar
        val result = Array(bmw,audi,benz)
        result
    }
}
// 调用
val luxurycars = LuxuryFactory.createcar()
for (x <- luxurycars){
    x.brand()
}

```

### syntax

- **方法覆盖的时候必须显式的写上override修饰符**，避免accidental overriding
-  **string interpolation**： val name="Tom"; val s = s" Hello ,$name"。在字符串前面加上“s”，就可以直接使用常量或者变量的值。
- **() is the fine representation of not having a value**. e.g. val x = ()
- **Tuple**: definition: ordered container of two or more values,there is not way to iterate through and change element in a tuple. elements in a tuple may have different data types. **访问tuple element 通过 t._index， index 从1 开始**
- 定义或者修改val，var的时候可以用multiple expression e.g. val test = { val x = 6; x +10}
- if-else 用于赋值 val x = if(a>b) a else b
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

### Immutable Collection

List

- **Immutable single linked list**
- val test = List(v1,v2,v3,v4...) elements 的type可以不同， list元素不可变
-  **Indexing**: L(index); L.head 返回首个元素， **L.tail 返回剩下的元素**
- **Nil** is singleton instance of List[Nothing], 可用于判断List 是否为空，和List.isEmpty 等价
- **List 1 ：：：List 2** 合并， 但是会去掉重复元素
- **List 1 ++ List2** , append操作 不会去掉重复元素
- **List:+ value** append操作， 由于list 是immutable的，所以不能直接用list + value 实现append， 必须需要“：”符号
- distinct 去重， filter 过滤， partition 按照规则把list分为两个tuple， reverse 反转；sortBy 按照规则排序 List.sortBy(_.size)

Set

- **immutable, unordered , a collection of unique elements**

Map

- val M: Map[keytype1 , valuetype] = Map()
- 使用“+ （key->value）" 来添加键值对，“- key” 来去掉键值对

### Mutable Collection

> 需要Import 下列package
>
> collection.mutable.Buffer
>
> collection.mutable.Set
>
> collection.mutable.Map
>
> val test = collection.mutable.Buffer(); test += value 就可以实现append操作

### Collection Function

**map**

  : 对RDD集合中的每个元素应用指定的function，一般来说，如果想实现一个for循环对一个iterable结构进行遍历执行某个操作，都可以用map代替。执行结果替代元素值, 

  值得注意一点就是原List 如果是不可变的类型的话，经过map function是不会改变原来的值的，如果需要保存结果就需要把结果赋值给其他变量


  ```
  val testList = List(1,2,3)
  testList.map(num => num*2)
  // => 符号表示映射
  out=List(2,4,6)
  ```
**foreach**

  : 主要用来遍历集合元素输出

  ```
  val testList = List(1,2,3,4)
  testList.foreach(num=>println(num))
  1
  2
  3
  4
  ```

**collectAsMap**

把[K,V]类型的RDD转换为Map格式，注意，**如果该RDD太大，会出现Java heap memory超的情况**

**flatten**

  对象是集合的集合， 把2层嵌套结构展平，超过两层就需要多调用几次，但是不如flatMap常用， stack overflow上说比flatMap more efficient （?）

  ```
  val testList = List(List(1, 2), List(3, 4), List(5, 6))  
  testList.flatten(num => num.tail)
  out: List(2,4,6)
  ```

**flatMap**

 和flatten差不多

  ```
  val testList = List(List(1, 2), List(3, 4), List(5, 6))  
  testList.flatten(num => num.map(num=>num*2))
  out: List(2,4,6,8,10,12)
  ```

**Join, leftOuterJoin, rightOuterJoin**

  ```
  #基本语法
  RDD.join(another RDD)
  只有Key , Value形式的RDD可以进行join, 也就是二元形式.如果一个RDD形如 (V1, V2, V3, V4)是不能join的
  可以转换为（V1,(V2,V3,V4)）的形式
  如果两个RDD形如 (v1, (v2,v3,v4)),  (v1,(v5,v6)) 两个RDD依据v1进行join操作之后的结果 R1.join(R2)
  (v1,((v2,v3,v4),(v5,v6)))
  ```

**reduce/reduceByKey**

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

### Rules

- object 定义的class 下如果包含main函数，则该object为singleton object，只有一个instance，且instance的名字和class的名字一样
- scala不存在静态成员（no matter methods or fields）scala使用Singleton中的成员来代替静态成员
- scala文件经过scalac编译之后生成的是标准的java class文件
- scala中everything is object，it is a pure object-oriented programming language. 数字是对象，也有自己的方法，1+2 实际上是（1）.+(2) 整数1调用他的“+”方法，传入的参数是2
- scala词法分析器采用long-match算法，最长匹配
- 函数的返回值可以implicitly defined， 根据函数体最后一行代码的类型自动判断
- 和动态语言不同，scala中if（0），if("") 会报错mismatch， 并不能automaticlly converted into boolean type.
- 尽管var 定义的变量值可以reassign at anytime。 但是var的类型是不能变的
- statement vs expression expression 有返回值， statement 没有返回值unit类型

### Option

option类通常作为scala集合类型（List，Map）的返回类型，option类型有两个子类，None和Some。当函数成功返回一个string的时候，回传some（string），如果程序返回为None，则没有得到想要的字符串，例如Map.get(key) key不在map中的时候返回None. 集合.get方法返回值不是具体的数值，或字符串，而是一个option结构，通过option.get方法获得具体的值.

 为什么要加option这个结构，把具体的返回值封装为一个结构，为了得到具体的值还需要多个get方法？ 大概是出于提醒人们处理NPE异常的目的，并且option结构后常常接flatMap，如果option为空，不执行flatMap，否则执行，省去一个判空的步骤。网上说的option结构解决NPE问题，实际不准确。如果实在想得到map中一个key对应的value值，

```
获得一个map中key对应的value具体值
Map.get(key).get
如果确定key对应的value不为null的话 可以直接写成
Map(key)
```

### SparkDataFrame

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

**Reference**<br>[scala 课堂](https://twitter.github.io/scala_school/zh_cn/index.html)<br>[Scala 函数柯里化(Currying)](https://www.runoob.com/scala/currying-functions.html)

## Scala 写SparkSQL

不多说，直接上代码

```scala
package  com.pack.path

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object scala_test {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("data").setMaster("local[*]")
    val sparkSession = 		SparkSession.builder().config(sparkConf).enableHiveSupport().getOrCreate()
    val rdd = sparkSession.sql("select * from tableA")
    rdd.show()
  }

```

pom文件要添加相应的依赖, 并且用mvn clean package指令打包，如果不使用插件，java -jar xxx.jar 并不能直接运行，因为在jar包META-INF/MANIFEST.MF文件中不存在main-class属性，程序不知道入口在哪里。插件的作用就是在打包的时候把我们指定的如果写入META-INF/MANIFEST.MF文件，这样jar包就可以直接用运行。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <!-- 项目名-->
    <artifactId>project_name</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <!-- 安装的scala的版本-->
        <scala-maven-plugin.version>3.1.3</scala-maven-plugin.version>
        <scala.version>2.13.3</scala.version>
    </properties>
    <dependencies>
        <dependency>
            <!-- spark的版本，需要和线上对齐 -->
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-core_2.11</artifactId>
            <version>2.2.0</version>
        </dependency>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-sql_2.11</artifactId>
            <version>2.2.0</version>
        </dependency>
        <dependency>
            <groupId>org.scala-lang</groupId>
            <artifactId>scala-library</artifactId>
            <version>${scala.version}</version>
        </dependency>
        <dependency>
            <groupId>org.scala-lang</groupId>
            <artifactId>scala-compiler</artifactId>
            <version>${scala.version}</version>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <!--插件，实现自定义打包，生成可直接执行的jar包-->
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-assembly-plugin</artifactId>
                <version>2.5.5</version>
                <configuration>
                    <archive>
                        <manifest>
                            <!-- main-class的名字package.classname -->
                            <mainClass>
                              com.pack.path.scala_test
                            </mainClass>
                        </manifest>
                    </archive>
                    <descriptorRefs>
                        <!-- 文件后缀，打包后jar包名字会添加上这个字符串 -->
                        <descriptorRef>jar-with-dependencies</descriptorRef>
                    </descriptorRefs>
                </configuration>
                <executions>
                    <execution>
                        <id>make-assembly</id>
                        <phase>package</phase>
                        <goals>
                            <goal>single</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>

            <plugin>
                <groupId>org.scala-tools</groupId>
                <artifactId>maven-scala-plugin</artifactId>
                <version>2.15.2</version>
                <executions>
                    <execution>
                        <goals>
                            <goal>compile</goal>
                            <goal>testCompile</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
```

如果一切顺利，现在已经在target文件夹下生成了相应的jar包了，对照spark-submit的参数把jar包提交spark环境运行即可

**Reference**<br>[Maven生成可以直接运行的jar包的多种方式](https://blog.csdn.net/xiao__gui/article/details/47341385)<br>[spark作业配置及spark-submit参数说明](https://blog.csdn.net/feng12345zi/article/details/80100317)

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

