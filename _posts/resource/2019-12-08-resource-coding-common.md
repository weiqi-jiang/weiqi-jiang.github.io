---
layout: post
title: Common
category: Resource
tags: coding common info
description: 一些编程语言通用的知识，例如正则表达式
---

# **编程语言通用知识**

## **命名规则**

### Summary

- 函数名，变量名，文件名要有描述性，少用缩写

### File Name

- all lower case
- may use "_" or "-"

### Class Name

- capitalized the first letter for every word
- do not use "_"

### Variable Name

- all lower case use "_" to connect words
- class variable names end with "_"
- structrue variables are treated as normal variables

### Method Name

- lower case the initial word and capitialize the follows words

### Parameter Name

- same as method's name

### Constant

- all upper case
- usually with "_"

### Function Name

- same as class name



## Basic Knowledge

**Byte**

8 bit(位) = 1 byte(字节)

-127 to 127

**Short**

16bit = 2 byte

-32768 to 32768

**Int**

32 bit = 4 byte

-21亿 to 21亿

**Long（在定义是需要加上L or \*l\*）**

64bit = 8 byte

数值太大，省略

**Float（1符号位+8指数位+23尾数位）**

32 bit = 4 byte

-3.4^38 to 3.4^38

**Double(1符号位+11指数位+52尾数位)**

64 bit = 8 byte

-2^1024 to 2^1024

符号位表正负，指数位表10的多少次方，尾数位表示小数点后的数字，整数部分始终隐含“1”; 参考文献：[float与double的范围和精度](https://www.cnblogs.com/BradMiller/archive/2010/11/25/1887945.html)

**Char**

16 bit =2 byte

### 常见转义字符

- \ '   单引号
- \\   反斜杠
- \t   相当于tab，移至下一个制表符
- \r   回车
- \n   换行
- \b   退格
- \f   换页

### 正则表达式

所有基本语法和常见用法参见： [正则表达式手册](https://tool.oschina.net/uploads/apidocs/jquery/regexp.html)

### C/C++ sizeof函数

常见基本类型的大小

​           32位         64位

char        1          1

int         4       大多数4，少数8

short       2          2

long        4          8

float        4          4

double      8          8

指针          4             8

对于结构体来说情况有点复杂，结构体的大小有两个原则：

1. **结构体变量中成员的偏移量必须是成员大小的整数倍（0被认为是任何数的整数倍）** 
2. **结构体大小必须是所有成员大小的整数倍，也即所有成员大小的公倍数。**

偏移量**指的是结构体变量中成员的地址和结构体变量地址的差**。结构体大小**等于最后一个成员的偏移量加上最后一个成员的大小**

```
struct stu1  
{  
     int i;  
     char c;  
     int j;  
}； 
# i的偏移量为0
# c的偏移量为4
# j的偏移量为5，但是偏移量要是成员大小的整数呗，所以漂移成8
# 最后整个结构的大小就是8+4 = 12
struct stu2  
{  
      int k;  
      short t;  
}；  
# k的偏移量为0
# t的偏移量为4
# 结构体的大小是6，但是需要符合条件2，所以偏移到8
```

**Reference**

[C/C++ sizeof函数解析——解决sizeof求结构体大小的问题](https://www.cnblogs.com/0201zcr/p/4789332.html)

### JSON 格式

**JavaScript Object Notation**

key 为字符串类型

对象结构：

{

key1： value1

key2： value2

}

数组结构

[

{

key1: value1

key2: value2

},

{

key3: value3

key4: value4

}

]

python 中可以想access dict一样 access JSON格式

### 小Tips

1.  int i = 'a'(单括号)  # 赋值‘a’ 对应的unicode值给 变量i； char i = 'a' 赋值char类型
2.  a 按位异或 b 得到c 进行加密 ； c 按位异或b 得到 a 进行解密
3.  System gc()  强制启动垃圾回收器



## UML类图

![uml](/assets/img/resource/common/uml.png)

![uml_interface](/assets/img/resource/common/uml_interface.png)

类图中的关系

1. Generalization :继承关系，表示一般到特殊；带三角箭头的实线，箭头指向父类
2.  Realization ：类于接口的关系；带三角箭头的虚线，箭头指向接口
3. Association: 一种拥有的关系，例如一个学生可以拥有很多门课程； 带普通箭头的实心线，指向被拥有者
4. Aggregation: 整体和部分的关系，且部分可以脱离整体单独存在，聚合关系是关联关系的一种； 带空心菱形的实心线，菱形指向整体
5. Composition: 整体和部分的关系，且部分不能脱离整体单独存在；带实心菱形的实线，菱形指向整体
6. Dependency: 依赖关系，表示一个类的实现需要另一个类的协助；带箭头的虚线，指向被使用者

![relations](/assets/img/resource/common/relations.png)

类的操作是针对类自身的操作，而不是它去操作人家。比如书这个类有上架下架的操作，是书自己被上架下架，不能因为上架下架是管理员的动作而把它放在管理员的操作里。

**Reference** 

[详解UML图之类图](http://www.uml.org.cn/oobject/201610282.asp)

在线画图[processon][https://www.processon.com/]