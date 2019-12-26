---
layout: post
title: Java
category: Resource
tags: java
description: 记录java常用语法，便于查看
---

# **Java**

### **Principle**

***java源程序source code*** 就是我们根据语言规范编写的程序，扩展名为java，使用java编译器生成***bytes code字节码\***，扩展名为class，编译器在JDK或JRE中，JDK包含JRE，**字节码不是二进制文件**，字节码文件通过java解释器在**java虚拟机**上解释执行，java虚拟机就是在机器和编译的程序之间的一层抽象机器，编译程序只需要面对虚拟机，由虚拟机中的解释器负责把编译代码解释成机器码，不同的机器有不同的解释规则。

java虚拟机就是JRE，包含在JDK中，虚拟机的存在使得source code 编译的字节码虽然在所有os上都是一样的，但是却可以在不同的机器上运行

**Java 堆内存 和栈内存**

堆空间用来存放new 创建的对象和数组，在堆中产生一个对象或者array后，在栈中定义一个引用变量，取值为object或者array的首地址，当程序运行到作用域以外时，引用变量释放，本身仍然存在堆内存中，当没有引用变量指向堆内存中的某个object或者array时，任然占据内存，在未来不确定的时间删除

函数中定义的变量和引用变量都在栈内存中，程序运行超过定义域时，自动释放内存

 

**Java中&& 和 &， || 和 | 都表示and 和 or**

区别在于 & 会比较两个表达式，而&& 在第一个表达式为false时，就直接输出false了， || 和 | 同理

 

print printf 和 printf 的区别

1. print 和println基本相同
2. printf 进行格式化输出； printf("i的值为%d, j 的值为%f"，i , j) ；%.2f 小数点保留后两位；%5d 表示输出5位int， 不足用0补位

### **Class 类**

**类的构造方法**

- 没有返回值
- 与类同名
- 不用void修饰

**静态方法**

- 静态方法内不可以直接调用非静态方法
- 不可以使用this关键词（this 关键词和python中self类似）
- 不能将method内的local variable设为静态

 

如果类中成员变量为定值 e.g.

public class example{

int c = 50;

}

实例化 e1,e2

e1.c = 60 实例化后的成员变量的操作不影响class中成员变量的值

但是如果class 中成员变量被设定为static，那么实例化后改变成员变量的操作会改变类中静态变量的值

**类的继承**

public class className extends superClassName

只支持完全继承和单继承

**类的隐藏**

当子类中声明与父类成员变量/静态方法一样名字时，父类对应的被隐藏

**类的覆盖（override）**

子类声明一个和父类相同的方法名，输入参数列表，返回值，权限，也就是除了方法体以外其他都一样的方法

覆盖时加上@override当注释，当没有@override时，编译器会报错

**类的重载（overload）**

子类和父类具有相同的方法名，权限的方法

**super关键词**

- 调用父类构造方法
- 可以理解为一个指向自己超类对象的指针，而这个超类指离自己最近的一个父类
- 操作被隐藏的成员变量，成员方法 super. 成员变量  super. 成员方法
- [java中this 和super的用法总结](https://www.cnblogs.com/hasse/p/5023392.html)

abstractr 关键词

- 用于创建抽象类，抽象方法
- e.g.  public abstract int Example();

**如果在方法中有和成员变量同名的局部变量时；以local variable 为主**

### **变量有效范围/能见度**

成员变量（member variable）

1.  instance variables
2. static variables 静态变量 可以跨类访问 通过className.staticVariable 访问，**静态变量和静态方法是为了提供数据共享或方法共享，例如PI 就可以被不同的类共用**

局部变量（local variable）

1. 局部变量可以和成员变量的名字相同
2. 成员变量在这时会被隐藏，通过className.staticVariable 访问隐藏的成员变量

### **可见度private，public,protected**

private var/method 只能在本类中使用，子类不可见

private class 隐藏所有数据

public var/method/class 本类，子类，其他包中类，本包其他类都可以访问

protected var/method/class 本类，子类，本包其他类

类的权限设置会约束类成员的权限

 

**这里是把类变量和static variable 搞混了，类变量不一定非要用static修饰，如果不用static修饰，实例对类变量的修改不会作用于类**

成员变量和静态变量的区别
1、两个变量的生命周期不同

成员变量随着对象的创建而存在，随着对象被回收而释放。

静态变量随着类的加载而存在，随着类的消失而消失。

2、调用方式不同

成员变量只能被对象调用。

**静态变量可以被对象调用，还可以被类名调用**。

3、别名不同

成员变量也称为实例变量。

静态变量也称为类变量。

4、数据存储位置不同

成员变量存储在堆内存的对象中，所以也叫对象的特有数据。

静态变量数据存储在方法区（共享数据区）的静态区，所以也叫对象的共享数据。

### **Syntax**

三元表达式： boolean b = 20<45? true: false;