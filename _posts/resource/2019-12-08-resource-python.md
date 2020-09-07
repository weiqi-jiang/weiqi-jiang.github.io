---
layout: post
title: Python
category: Resource
tags: python
description: 主要是记录一些平时不经常用，但是偶尔还挺有用的python语法
---

> python作为一个有点“骚”的语言, 是值得多学习的~

## 1 基础

### 1.1 编码解码

python3 默认编码模式是utf-8，python2默认编码模式是ASCII，如果想指定编码格式

```python
# -*- coding: windows-1252 _*_

# ord 返回string 对应的unicode code
ord('a')
# chr返回unicode code对应的string
chr(97)
```

### 1.2 生成器/迭代器

```python
''' 
为什么要使用生成器？ 
因为如果一次性把list全部计算保存在内存中，如果list大小不大的话，情况还行，但是如果list很大的话，会消耗很大的内存容量，但是生成器不存在这个弊端，它只是存储一个计算方式，访问的时候再计算，不会占据很大的内存容量
凡是可以用作next()的对象都是iterator， generator属于iterator
对象实现了__iter__方法是可迭代的，实现了next()方法的是迭代器
'''
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b

    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己
    
    def next(self):
        self.a, self.b = self.b, self.a + self.b # 计算下一个值
        if self.a > 100000: # 退出循环的条件
            raise StopIteration();
        return self.a # 返回下一个值

# 形成generator最简单的方式是用列表,字典，set生成式，只是需要把[], {} 修改为()
(i for i in range(10))

# 函数中包含yield关键词，该函数就变成了一个生成器
def g(x):
	while x <=10:
		yield x 
		print('hello world')
		x += 1 
		
# 可以把生成器当成一个iterable list，可以直接list(g(3))	
for i in g(5):
	print(i)
	
'''
next 返回生成器的下一个输出值，这里有一个点需要注意，第一次next不会输出hello world，执行到yield语句返回3，就停住了，相关的信息记录在yield(), next()去重新读取,用书中的原话来说是“yield” pauses a function.“next()”resumes where it left off. 
下一次执行next的时候才会输出hello world，然后进入for loop 返回4再次停住,当生成器中没有元素，会抛出StopIteration Exception
'''
```

### 1.3 assert 语句

```python
'''
如果assert 后面合法的python语句是True, assert do nothing 
如果assert 不成立，抛出AssertionError Exception ，可以自定义错误信息

'''
assert 1 + 1 ==2, 'error message' 
```

### 1.4 原生方法

**\_\_new\_\_ /\_\_init\_\_**

```python
"""
__init__是当实例对象创建完成后被调用的，然后设置对象属性的一些初始值。
__new__ 第一个参数是这个类，__init__第一个参数是这个类的实例对象
__new__是在实例创建之前被调用的，因为它的任务就是创建实例然后返回该实例，是个静态方法。
即__new__在__init__之前被调用，__new__的返回值（实例）将传递给__init__方法的第一个参数，然后__init__给这个实例设置一些参数。
__init__ 并不是其他语言中常说的构造函数，而是初始化函数，因为在调用init之前已经由new构造出了一个实例
"""
class foo:
    def __new__(cls,*argv,**kwargv):
        # 第一个参数是这个类，其余参数在调用成功后传递给__init__方法,默认是调用超类的__new__方法
        # __new__ 方法的作用是以合适的参数调用超类的 __new__ 方法
        return super.__new__(cls, *argv,**kwargv)
```

**\_\_name\_\_**

模块也是一个对象，该对象有\_\_name\_\_ 属性

```python
import xxx_module
# 输出结果是模块文件名，没有路径和文件名
xxx_module.__name__

# python xxx_module.py 如果直接运行模块，__name__ 是默认值__main__

```

**\_\_repr\_\_**

```python
#每个类都有一个__repr__方法，自定义print 实例化对象时的返回值，默认的输出是类名+object at + 内存地址
class test:
    def __init__(self):
        pass
	def __repr__(self):
    	return 'xxx'
print(test()) # 输出xxx
```

**\_\_del\_\_**

```python
#手动或者自动释放空间的时候，会调用__del__()方法,但是要注意不要随意重载，确保资源能够正确释放
class test:
    def __init__(self):
        pass
	def __del__(self):
    	return 'xxx'
```

**\_\_dir\_\_  / \_\_dict\_\_**

dir 返回对象拥有的所有方法和属性, dict 查看属性名和属性值组成的字典

**\_\_call\_\_**

相当于在类中重载‘()’运算符，使得**类实例**对象变成可调用对象。python中可以将“()”应用到本身执行，都称为“可调用对象”, 一般的情况下，新建类实例时，调用\_\_init\_\_方法, 但新建的实例本身并不能直接调用，如果执行，或报错说“xxx object is not callable”。但是如果在类定义中实现了\_\_call\_\_方法，则可以把类实例变成可调用对象。

```python
class foo:
    def __init__(self)：
    	print("111")
    def __call__(self):
        print("222")
        
a = foo()
# 111 
a()
# 222
```

**getattr(), setattr(), hasattr()**

```python
hasattr(obj, name)# 属性和方法都属于attr，返回True false， 无法分清属性还是方法
getattr(obj, name[, default]) # 返回属性值， 或者方法信息，如果属性或方法不存在对象中，返回default，如果没有指定默认值，抛出AttributeError
setattr(obj, name, value)
```

**Reference**<br>[Python \_\_call\_\_()方法（详解版）](http://c.biancheng.net/view/2380.html)<br>[通俗的讲解Python中的\_\_new\_\_()方法](https://blog.csdn.net/sj2050/article/details/81172022)<br>[Python \_\_new\_\_()方法详解](http://c.biancheng.net/view/5484.html)

### 1.5 命名方式

```python
object #公用方法

"""
半保护,被看作是“protect”，意思是只有类对象和子类对象自己能访问到这些变量
在模块或类外不可以使用，不能用’from module import *’导入。
"""
_object 

"""
全私有，全保护,私有成员“private”，意思是只有类对象自己能访问，
连子类对象也不能访问到这个数据，不能用’from module import *’导入
__object 也是为了避免与子类的属性或方法名称冲突， 对于该标识符描述的方法，父类的方法不能轻易地被子类的方法覆盖，他们的名字实际上是_classname__methodname
"""
__object

"""
内建方法，用户不要这样定义
"""
__object__ 

"""
当想要强行使用关键词作为变量名的时候，后面加一个下划线用作区分
"""
object_
```

### 1.6 转义字符

python里面%d表数字，%s表示字符串，%%表示一个%；单引号内嵌套单引号需要转义字符/;单引号内嵌套双引号不需要嵌套；双引号内嵌套双引号需要转义字符/；双引号内引用单引号不需要转义字符；

### 1.7 set

相比于list 和tuple，set相对不常用

```python
s = set()
s.add(1)
# 都会把set list中的每次元素执行一次add操作，如果元素已经存在于set，不会报错
s.update({3,4,5},{6,3,4})
s.update([2,5,7])
# 如果discard元素存在set中，将之移除，如果不存在不会报错;remove如果不存在会报错
s.discard(10)
s.remove(10)
# 清空
s.clear()
# set也有pop方法，但是是无序的,不在于输出顺序的时候可以用
s.pop()

# 两个set的交并补
s1 = set()
s2 = set()
#并集
s1.union(s2)
# 交集
s1.intersection(s2)
# 在s1不在s2的元素
s1.difference(s2)
# 返回只出现在一个set中的元素
s1.symmetric_diffence(s2)

# 两个set的关系
# s1是否是s2的子集
s1.issubset(s2)
# s1是否是s2的父集
s1.issuperset(s2)
# 另一种方式判断包含关系
a = [1,2,3,4]
b = [1,2,3]
c = [1,2,3,5] 
a = set(a)
b = set(b)
c = set(c) 
a>b # True
a>c # False 
```

### 1.8 dict

判断输入是否存在于key set中

```python
# O(n)
if key in dict.keys()
# O(1) 
if dict.get(key)
```

### 1.8 浮点数精度

```
浮点数的小数部分转换为二进制小数的时候都是不断的*2 取整数部分的值作为一个二进制位0,1然后对剩余的小数部分重复操作
0.1*2 = 0.2  整数0，小数0.2
0.2*2 = 0.4 整数0，小数0.4
0.4*2 = 0.8 整数0，小数0.8
0.8*2 = 1.6 整数1 小数0.6
0.6*2 = 1.2 整数2 小数0.2
重此开始了重复,但是计算中不能无限循环，保存精度有限，所以python默认精度有17位，16位精准，17开始不精准 
```

### 1.9 异常处理

```
try:  
	do sth0
except KeyError e:
	do sth1
else:  
	do sth2
finally:  
	do sth3
```

在没有return 的情况下：try首先执行，如有异常跳到except，如果没有执行else，finally是一直要执行的有return的情况下： 不管怎样，finally部分的代码是一定执行的，所以finally中有return的话，就按照finally 执行后的结果return 即使try部分，except，else 部分有return ，也是要先执行finally，finally没有return 就返回去return； 如果有就在finally 处return 了，不会回到原来的地方。所以结果可能和预计的不太一样。手动抛出异常raise ExceptionType('description')，常用的ExceptionType

- AssertionError
- AttributeError
- IndexError
- KeyError
- ValueError
- SyntaxError
- TypeError
- ZeroDivisionError
- NameError

手动抛出异常常常不是用来让程序停下来，而是反过来，让程序能够顺利执行，因为raise经常用在try，exception语法的try语句中，当条件不满足时，例如用户输入类型错误，程序本身不会报错，但是结果会错误，所以故意抛出异常触发exception语句

### 1.10 self解释

1. self代表是类的实例，而不是类本身，self.__class__ 才是类本身
2. self只是一个约定俗成的写法，本身就是一个参数，所以是可以更改的，比如写成this
3. **self是不能不写的**，比如有一个成员函数是Test类的成员函数是test，实例是a，解释器运行的时候是Test.test(a)把实例当成一个参数传入到self的位置，所以self是不能不写的。有一种情况可以不写，那就是类方法，只能通过Test.test()这种方式去调用。

**Reference**<br>[一篇文章让你彻底搞清楚Python中self的含义](https://www.cnblogs.com/jessonluo/p/4717140.html)

### 1.11 Path相关

首先是从sys.path下搜索

```python
import sys
# 返回一个系统路径string 的list，可以像操作任何list一样操作sys.path
sys.path
# 添加搜索路径，只在当前python程序运行时间内生效
sys.insert(0, 'path')


import os
# 返回current work directory
os.getcwd()
#改变工作目录,支持相对于当前工作路径的相对路径
os.chdir(path)
# 展示路径下所有文件名
os.listdir(path)
# 遍历路径下所有文件
for file in os.listdir(path):
    filepath = "{0}/{1}".format(path, file)
# 判断文件是否存在或路径是否存在

# 路径拼接
os.path.join(path1, path2)
# split会把路径分成目录路径和文件名
dirpath, filename = os.path.split(path)
# 把文件名和扩展名分开
filename, extension = os.path.splitext(filename)

# 内置的glob模块，给定通配符，返回符合条件的文件名
import glob
glob('dirpath/*.py')

```

### 1.12 取整方式

1. int(x); 直接抛去小数部分，保留整数部分，正负数皆如此
2. round(x); 四舍五入
3. math.ceil(x) 上取整

### 1.13 文件

```python
# 输出会在test.txt原来文本的基础上多一个空行，因为read()在文件结尾处返回一个空字符串显示出来就是空行。可以使用rstrip删除掉
with open('test.txt') as file:
    text = file.read()
    print(text)
    
# 各行内容列表
with open('test.txt') as file:
    lines = file.readlines()
for line in lines:
    print(line)

# 分行读取
with open('test.txt') as file:
    for line in file:
        print(line)

# 和input一样，读取文件也会把所有文本都解读为字符串，所以数字需要类型转换

# 写入, 可选模式有‘r’, 'w', 'a', 'r+', 'w+'，‘a+’
# r只读， w新建只写，w+新建读写 都会把原来的文本清空， r+读写不新建，a附加写，不可读，a+附加读写 
# 写入只能写入字符串，需要提前转换， write不会自动添加换行符，所以多个write之间会连在一起
with open('test.txt', 'w') as file:
    file.write('xxxxxx')
```

### 1.14 编程规范

1. 先import标准库模块，再添加一个空行，再import自己编写的模块

### 1.15 py2/py3区别

1. python2 math.floor(5.5) 返回5.0 python3 math.floor(5.5) 返回5

## 2 进阶

### 2.1 抽象类

抽象类的个人理解一部分作用是函数名规范化，例如很多个线性模型，都有fit方法和predict方法，如果模型不继承抽象类，fit方法可能因为不同的程序员开发变成不同的名字，比如fit_data之类，可能会导致各种事先没有考虑到的错误。如果提取出抽象类，各个线性模型都继承抽象类，则必须实现抽象类中的fit和predict方法，相当于规范了函数名。python中没有接口，但是python支持多继承，所以在需要多继承场景下，直接继承多个抽象类即可。

```python
from abc import abstractmethod, ABCMeta


class LinearModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        pass 
    
    @abstractmethod
    def predict(self):
        pass 
   	#非抽象方法
	def print(self):
        print('hello world')
```

### 2.2 装饰器

装饰器的本质还是函数或者类，它的作用是在其他python函数外套上一个“壳”，在保留原来功能的基础上，添加新的功能，比如写日志，性能测试, 甚至只是为了“装饰”...

```python
##########被装饰函数无参 #################
def logging(func):
    def wrapper():
        print("this is wrapper function")
        return func()
    return wrapper 

# 不使用@语法糖,看起来更明了，相当于用新函数覆盖了原来的函数名
def foo():
	print("hello world")
foo = logging(foo)

# 使用@语法糖
@logging
def foo():
    print("hello world")
 
foo() # 打印this is wrapper function 然后打印hello world

############ 被装饰函数有参 ################
def logging1(func):
    """一个参数"""
	def wrapper(param):
        print("this is wrapper function")
        return func(param)
    return wrapper

def logging2(func):
    """适配所有情况"""
    def wrapper(*args, **kwargs):
        print("this is wrapper function")
       	return func(*args, **kwargs)
   	return wrapper

@logging2
def foo(a1,a2,a3=0,a4=0):
    print("hello world")
    
foo(1,2,a3=1,a4=2)

############ 装饰器带参数 ################
#看起来很复杂，套了3层函数，其实很简单就是在原来无参装饰器的基础上再套一层方便传level参数而已
# 所以写法可以先按正常不带参数装饰器写，最后再套一层加个参数。
def logging(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level=='1':
            	print("Decorator with parameters level-1")
            elif level=='2':
                print("Decorator with parameters level-2")
            else:
                print("Decorator with parameters level-3")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@logging(level='1')
def foo(a):
	print("hello world")

foo(1)
# Decorator with parameters level-1
# hello world

############# 其他 #####################
def logging(func):
    print("Testing")
    def wrapper():
        print("this is wrapper function")
        return func()
    return wrapper 

@logging
def foo():
    print("hello world")

"""
输出： 
Testing
为什么？看起来没有调用函数却有打印：@logging -> foo = logging(foo) 这个时候已经调用了logging函数
"""

foo()
foo()
"""
输出：
this is wrapper function
hello world
this is wrapper function
hello world
为什么又没有Testing输出了呢：因为之前有过foo = logging(foo)，所以foo()已经相当于调用wrapper()
print("Testing") 语句在wrapper定义外，当然不会打印，那随之引出一个问题，如果我调用两次@logging呢？
会不会打印两次Testing，会！
"""
@logging
@logging
def foo():
    print("hello world")
"""
直接运行 输出两次“Testing”，接着又引出另一个问题，我如果调用foo(),会打印什么？
"""
foo()
""" 输出：
this is wrapper function
this is wrapper function
hello world
此时相当于 foo = logging(logging(foo)) 调用顺序是从内到外
"""

##################类装饰器#################
# step 1
class decorator:
    def __init__(self, func):
        self._func = func 
        print('in initilizer')
        
    def __call__(self):
        print("in decorator")
        self._func()
# step 2
@decorator  
# step 3
def foo():
    print("111")
"""
在step2时，已经打印出了in initilizer, 说明类装饰器是先初始类实例， 相当于foo = decorator(foo)
此时foo已经是decorator类的实例，调用foo变成了调用类实例，所以一定要定义__call__方法，且__call__方法中一定要调用self._func
"""
```

**Reference**<br>[理解python装饰器看这一篇就够了](https://foofish.net/python-decorator.html)

### 2.3 单例模式

```python
########### 函数装饰器方法 ###########
"""
为什么可行？
@singleton -> foo = singleton(foo),此时foo() 相当于inner()
_instance中如果有类地址的键（注意！类地址是不会变的！），返回之前创建的实例，如果没有，创建实例
问题？
_instance 的作用域是多大？ foo._instance报错，说明不是类变量，但是inner() 函数又可以访问
"""
def singleton(cls):
    """singleton decorator"""
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


@singleton
class foo:
    def __init__(self):
        pass 

c1 = foo()
c2 = foo()
c1==c2 #True

######### 类装饰器方法 #############
# 如果理解类装饰器的原理很好理解为什么这个方法是有效的，因为此时foo已经是singleton类的一个实例
# foo(*args, **kwargs) 其实相当于s = singleton();s(*args, **kwargs)
class singleton:
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]
    
@singleton
class foo:
    def __init__(self,*args, **kwargs):
        pass 
f1 = foo(1,2)
f2 = foo(1,2)
f1==f2
########### 借助__new__ 方法###########
# 通过覆写__new__方法，在实例化的时候进行控制来达到单例目的
# 注意的是__new__方法和__init__ 方法除了第一个参数以外其他参数要一致，原因在上文原生方法中提到
class singleton():
    _instance = None
    def __init__(self, *argv,**kwargvs):
        pass
    def __new__(cls, *argv,**kwargvs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance
    
s1 = singleton(1,2)
s2 = singleton(1,2)
s1 == s2

```

## 3 常用Module

### 3.1 sys Module

**指定编码格式**

python2版本默认的编码模式是Ascii，程序中如果出现非ascii编码字符，会报错例如：UnicodeDecodeError: 'ascii' codec can't decode byte 0x??如果想在python2版本中指定编码格式

```python
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# sys.getdefaultencoding() 返回现在默认的编码格式

'''
当module 第一次被import的时候会运行模块代码并加载到内存，多次重复import不会重复运行只会把该模块的内存地址引用到本地变量环境, reload 函数强制重新载入之前载入过的module，且module 之前必须被import过。
'''
```

而在python3.4以上的版本，reload函数被放在了importlib标准库中

```python
import importlib
importlib.reload(sys)
```

并且在python3版本中，默认的编码方式就是'utf-8', 并且python3没有sys.setdefaultencoding()函数，只有sys.getdefaultencoding()

**Reference**<br>[Python 解决 ：NameError: name 'reload' is not defined 问题](https://blog.csdn.net/github_35160620/article/details/52206868)<br>[python为什么需要reload(sys)后设置编码](https://www.cnblogs.com/fengff/p/8857360.html)

### 3.2 logging Module

```python
import logging 
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'，filename = "logs")#运行时间，模块名字，级别名字，信息 
logger = logging.getLogger(__name__) # 传入模块的名字 
logger.info("xxx") #输出info级别的信息 
logger.debug("xxx") # 文件级别是info的话 debug信息是不会输出的logger.warning("xxx") |
```

### 3.3 argumentParser Module

```python
import argparse 
parser = argparse.ArgumentParser() #ArgumentParser的参数都是keyword参数

# optional argument "--" prefix sign for optional argument 
# argument without "--" or "-" prefix is the locational argument 
parser.add_argument("-a , --argname") 

parser.parse_args() # or parser.parse_known_args() 
```

### 3.4 numpy Module

**reshape(-1,1)和reshape(1,-1)**

```python
>>> a = np.array([[1],[2],[3]])
>>> a.reshape(-1,1)
array([[1],
       [2],
       [3]])
>>> a.reshape(1,-1)
array([[1, 2, 3]])
```

### 3.5 JSON Module

```python
import json
# json dump 接受两个实参，数据和可存储数据的文件对象
with open('test.json', 'w') as file:
    data = [1,2,3]
	json.dump(data, file)
    # 当然这里存储的数据不是正规的json格式，但是不影响存储，依然可存可读

# json load
with open('test.json', 'r') as file:
    data = json.load(file)
```

## 4 杂七杂八

### 4.1 打包
```python
#在项目文件顶层文件夹下新建setup.py 文件,并且在想打包进package的文件夹下加上名为__init__.py的空文件

# setup文件实例
from setuptools import setup 
setup( name = "modulename"， version = "1.0", packages = ["src"] ) 

#完成setup.py文件 之后需要运行脚本生成package
#创建egg包
python setup.py bdist_egg
#创建tar.gz 包
python setup.py sdist --formats=gztar 

#之后会在项目文件夹下生成新的几个文件夹，其中打包完的文件放在dist文件夹下上传tar包之前一定检查一下是不是包里包含所有需要的东西

# 打包成exe, -F 表示打包成单一exe文件， -p表示其他依赖文件
pyinstaller -F xxxx.py -p xxxx.py
```
### 4.2 执行脚本

```python
#pythonfile1.py: 
from time import sleep 
sleep(10) 

#pythonfile2.py 
print('python file 2')  

#python file 3: 
import os 
cmd = "python pythonfile1.py" 
status = os.system(cmd) # 如果上面cmd成功执行返回0 否则...不同情况返回不同的值 
if status ==0:  
	cmd = 'python pythonfile2.py' 
	os.system(cmd) 

# 运行的时候会先sleep 10秒 然后在打印 python file 2. 说明会先等到上一个脚本的执行结果的返回才运行下一个cmd
# 但是也有特殊情况 如果上一个cmd是用于提交cloud的指令，是不会等待cloud 运行完才返回；只要成功提交了cloud任务，cmd就会成功返回。
```

### 4.4 变量赋值引用

```python
res = []

#不会创建副本
res.append(num) 

#如果res是可变的 不会创建副本，如果是不可变的，会创建新的变量
res+=[num] 

#会创建副本
new = res + [nums] 
```

### 4.6 自定义排序函数
```python
# 第一种方式
def func(x,y):  
	if x<y:   
		return -1  
	if x==y:    
		return 0 
	if x>y:    
		return 1 

unsorted = [3,6,2,8,4]
sorted( unsorted, func)# 第二个参数是类型是func 所以不需要传递func（x，y） 

# 第二种方式 lambda expression sorted( unsorted, lambda x: len(x) ) 
```

### 4.7 datetime&time&string

```python
import datetime

# 字符串
st = '2017-11-23 16:10:20'

# 当前时间戳
dt = datetime.datetime.now() 

# datetime to string
dt.strftime("%Y-%m-%d %H:%M:%S")

# string to datetime
datetime.datetime.strptime(st, '%Y-%m-%d %H:%M:%S')

# string to timestamp
time.mktime(time.strptime(st,'%Y-%m-%d %H:%M:%S' ))

# timestamp to string
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sp))

# dayofweek
dt.weekday()

# 日期差值转int天数
(dt1-dt2).dt.days
```

### 4.8 os交互

```python
import os 

# 访问当前路径下文件名,不包含子目录中的文件; 返回一个文件名列表
for file in os.listdir(path):  
	print(file) 

# 和listdir 不同的是 walk 返回一个迭代器,通常通过for loop访问
for root, dirs, files in os.walk(file_dir):  
	print(root) # 当前目录路径  
	print(dirs) # 当前路径下所有子目录  
	print(files) # 当前路径下所有非目录子文件

```

### 4.9 跨级引用

需要把上一级module 的地址加入到系统路径中才能正常引用

```python
import sys
sys.path.append('..')
```
