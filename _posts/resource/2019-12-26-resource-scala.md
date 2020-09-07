---
layout: post
title: Scala
category: Resource
tags: scala
description: 
---

## Scala

[官方文档](https://docs.scala-lang.org/zh-cn/)

### Base

1. print和println的区别在于print 不会再内容后追加换行符，println会追加换行符

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

### Var

**val 不可变指的是不能改变初始化指向的类的实例对象，但是对象的值可以变；var可变指的是可以改变初始化指向的类的实例对象，而且实例对象自己也可以变**。关于var变量的赋值有一个和python大不一样的地方，简单理解就是除非新声明变量，否则不能同时给多个变量赋值. 但如果一个函数需要返回不同的数据类型的一组值，推荐新建一个类，类中包含相应的字段(典型的java作风)

```scala
//al，var都必须要定义时赋值，var可以使用默认初始化,但是必须指定数据类型，否则报错, ()表示没有值
//var 虽然值可以变,但是类型不能变
var a:String = _  //初始为null
var a:Int = _ // 初始为0 
val a = ()

// 利用tuple同时给多个变量赋值,
val (a, b) = (1, 2)

//函数返回值赋值，首先定义一个函数返回一个(Int,Int)的tuple
def t={(1,2)}
// condition 1 可以成功赋值
val (a,b) = t
// condition 2 会报错 error";" expected but "=" found 
var c = 0
var d = 0
(c, d) = t 
// 会报错
val(e, f) = (1, 1)
(c, d) = (e,f)
//分开赋值就没有问题
c = e
d = f
c=e; d=f;
```

当val 用**lazy** 修饰时，初始化被推迟直到第一次取值

```scala
lazy val a = 0
```

Reference<br>[scala学习手记5 - 元组与多重赋值](https://www.cnblogs.com/amunote/p/5559867.html)

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

字符串变量替换, 在字符串前面加“s”可以使用变量，加“f”可以格式化输出

```scala
val name = "jack"
// 加s可以直接使用变量名
print(s" my name is $name ")
// 可以在{}中写任何表达式
print(s" my name is ${name + "xxx" }")
// printf格式化输出，字符串前面加f
val score = 0.5
printf(f"score is $score%.2f")
```

**Reference**<br>[scala字符串前加s使用$](https://www.cnblogs.com/pursue339/p/10619581.html)

### Function

**返回值**，普通函数不明显使用return指明返回值，如果有返回值，最后一行就是返回值，**除了循环没有返回值，其他都有返回值**。**为什么不显式使用return**？，首先因为scala是函数式编程语言，所有东西都是表达式不是语句，一个表达式的最终结果就是想要的值，不需要写return，其次如果显式使用return，程序在执行到return时，强制退出方法，使得返回值类型推演失效，需要手动指定。

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

// 但是递归函数必须显式指定返回值
def factorial(n: BigInt): BigInt = {  
    if (n <= 1)
    	1
    else    
    	n * factorial(n - 1)
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

**Reference**<br>[scala泛型](https://fangjian0423.github.io/2015/06/07/scala-generic/)<br>[scala 课堂](https://twitter.github.io/scala_school/zh_cn/index.html)<br>[Scala 函数柯里化(Currying)](https://www.runoob.com/scala/currying-functions.html)<br>[scala中的函数哪些有返回值,哪些没有返回值??](https://blog.csdn.net/u010916338/article/details/77585213)

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

**Break**

```scala
// 虽然可以实现， 但是不推荐使用，可以用while(flag) loop 修改flag的值来实现break
import scala.util.control.Breaks._
breakable{
    for(i<-Range(0,10)){
        if(i>5) break
        println(i)
    }
}
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

**抽象类** 用**abstract** 关键词修饰，定义了一些方法但是没有实现，且不可被实例化,**抽象类的作用**(个人理解)： 1. 规范化，继承抽象类的子类拥有共同的方法名，不同的开发人员参考同一个规范。2.统一数据类型，比如在简单工厂模式中根据输入实例化不同的类，这些不同的类具有公共的数据类型, 继承抽象类的**子类必须实现抽象类的所有方法** 3.代码复用，抽象类中可以定义非抽象方法，如果这个方法有高度的通用性，在子类中则不用重复实现。

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

**Array** 有序，可变，包含重复项, **定长**， 元素类型可以不同

```scala
// array声明, 提供初始值时不要用new关键词，使用new关键词修饰时，初始为null
val numbers = Array(1,2,3,4,5)
val numbers = new Array[Int](10)

// array 元素访问,head返回头元素，tail返回剩下的元素而不是尾元素
val n3 = numbers(3)
val h = numbers.head
val r = numbers.tail

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

// sortBy 按照某个规则排序
a.sortBy(_.size)

// 反转、去重
a.distinct
a.reverse
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
m3 += ("k7"->"v7")

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

**List** 有序，元素不可变，可包含重复项

```scala
// 声明 和Array的区别在于 List在底层实现是一个链表，而且元素不可变
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

### Collection Function

**map**<br>对RDD集合中的每个元素应用指定的function，一般来说，如果想实现一个for循环对一个iterable结构进行遍历执行某个操作，都可以用map代替。执行结果替代元素值, 值得注意一点就是原List 如果是不可变的类型的话，经过map function是不会改变原来的值的，如果需要保存结果就需要把结果赋值给其他变量


  ```scala
 val testList = List(1,2,3)
 val out = testList.map(num => num*2)
  ```

**foreach**<br>主要用来遍历集合元素输出

  ```scala
 val testList = List(1,2,3,4)
 testList.foreach(num=>println(num))
/*
输出
  1
  2
  3
  4
 */
  ```

**collectAsMap**<br>把[K,V]类型的RDD转换为Map格式，注意，**如果该RDD太大，会出现Java heap memory超的情况**

**flatten**<br>对象是集合的集合,把2层嵌套结构展平，超过两层就需要多调用几次，但是不如flatMap常用， stack overflow上说比flatMap more efficient （?）

  ```scala
val testList = List(List(1, 2), List(3, 4), List(5, 6))  
testList.flatten(num => num.tail)
// out: List(2,4,6)
  ```

**flatMap**<br>和flatten差不多

  ```scala
val testList = List(List(1, 2), List(3, 4), List(5, 6))  
testList.flatten(num => num.map(num=>num*2))
//out: List(2,4,6,8,10,12)
  ```

**Join, leftOuterJoin, rightOuterJoin**<br>基本语法RDD.join(another RDD)只有Key , Value形式的RDD可以进行join, 也就是二元形式.如果一个RDD形如 (V1, V2, V3, V4)是不能join的可以转换为（V1,(V2,V3,V4)）的形式如果两个RDD形如 (v1, (v2,v3,v4)),  (v1,(v5,v6)) 两个RDD依据v1进行join操作之后的结果 (v1,((v2,v3,v4),(v5,v6)))

**reduce/reduceByKey**<br>reduce把RDD 的两两元素传递给操作函数，返回一个和元素同类型的值，然后和下一个element进行同样的操作，直到最后一个值。例子：

求和：.reduce(\_+\_) 
求最大值： .reduce( (a,b)=> if(a>b) a else b ) 
集合对应位置相加： .reduce((a,b) => (a._1+b._1, a._2+b._2, a._3+b._3))

reduceByKey 对象是key-value类型的RDD，返回值也是RDD类型，如果是3元及以上的RDD，需要转换为二元key-value 例如（1,2,3,4）不能直接reduceByKey，先转换为（1，（2,3,4））,"\_"占位符代表是value元素 .reduceByKey(\_+\_)

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

spark 中的dataframe 和RDD一样也是一个分布式的存储结构，并不是pandas中dataframe in memory 的数据结构[详细对比](http://www.lining0806.com/spark%e4%b8%8epandas%e4%b8%addataframe%e5%af%b9%e6%af%94/)

```
# pandas dataframe to spark dataframe
SQLContext.createDataFrame(pandas_df) 

# spark dataframe to pandas dataframe 需要保证spark_df 很小，因为pandas_df 不是分布式的结构，需要全部加载进内存的
pandas_df = spark_df.toPandas() 

# spark.dataframe 虽然是分布式存储的，但是可以显示的指明加载到内存
# 虽然全部加载到内存，但是类型还是spark.dataframe
# SQLContext.sql('''xxx''')的返回值就是spark.dataframe类型
spark_df.persist() / spark_df.cache()  
```

debug:
在pandas.dataframe 转成spark.dataframe 的时候可能会有‘Can not merge type <xxxxx>’
解决方法： df中存在空值，需要先处理空值，处理完可能还是不行，这个时候就需要强制类型转换，强制保证一个字段下数据的类型一致



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
    OfflinePath = "path"
    articleInfoPath = "path"
    id = 10
  }
  ```

  上述代码为.conf文件示例代码， 值得注意一点是 **字段前不要加 val 关键字**



