---
layout: post
title: Python
category: Resource
tags: python
description: 主要是记录一些平时不经常用，但是偶尔还挺有用的python语法
---

### python作为一个有点“骚”的语言, 是值得多学习的~



#### 打包，用于提交到cloudml分布式平台上运行
```
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
```
#### 执行其他脚本或cmd command

```
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
# 但是也有特殊情况 如果上一个cmd是用于提交cloudml的指令，是不会等待cloudml 运行完才返回；只要成功提交了cloudml任务，cmd就会成功返回。
```



####  logging module:  用于统一方便的管理日志
```
import logging 
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'，filename = "logs")#运行时间，模块名字，级别名字，信息 
logger = logging.getLogger(__name__) # 传入模块的名字 
logger.info("xxx") #输出info级别的信息 
logger.debug("xxx") # 文件级别是info的话 debug信息是不会输出的logger.warning("xxx") |
```


#### ArgumentParser
```
import argparse 
parser = argparse.ArgumentParser() #ArgumentParser的参数都是keyword参数

# optional argument "--" prefix sign for optional argument 
# argument without "--" or "-" prefix is the locational argument 
parser.add_argument("-a , --argname") 

parser.parse_args() # or parser.parse_known_args() 
```


#### 判断两个列表之间的包含关系
```
a = [1,2,3,4]
b = [1,2,3]
c = [1,2,3,5] 
a = set(a)
b = set(b)
c = set(c) 
a>b => True
a>c => False 
```



#### 变量赋值引用
```
res = []

#不会创建副本
res.append(num) 

#如果res是可变的 不会创建副本，如果是不可变的，会创建新的变量
res+=[num] 

#会创建副本
new = res + [nums] 
```


#### 检查输入是否在dict的key set中
```
# O(n)
if key in dict.keys()
# O(1) 
if dict.get(key)
```


#### 自定义排序函数；即自定义“大小”的定义
```
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


#### char转ASCII CODE； ASCII CODE 转字符
```
s = 'C'ord(s) # return ASCII 码
chr(ord(s)) ASCII码转回char 

```


#### 浮点数精度问题
```
浮点数的小数部分转换为二进制小数的时候都是不断的*2 取整数部分的值作为一个二进制位0,1然后对剩余的小数部分重复操作
0.1*2 = 0.2  整数0，小数0.2
0.2*2 = 0.4 整数0，小数0.4
0.4*2 = 0.8 整数0，小数0.8
0.8*2 = 1.6 整数1 小数0.6
0.6*2 = 1.2 整数2 小数0.2
重此开始了重复,但是计算中不能无限循环，保存精度有限，所以python默认精度有17位，16位精准，17开始不精准 
```



#### datetime & time & string 相互转换, 已经常用的指令
```
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



#### 与os 的一些交互
```
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

 

#### __new__ 和__init__的关系

- __init__是当实例对象创建完成后被调用的，然后设置对象属性的一些初始值。
- __new__是在实例创建之前被调用的，因为它的任务就是创建实例然后返回该实例，是个静态方法。

即，__new__在__init__之前被调用，__new__的返回值（实例）将传递给__init__方法的第一个参数，然后__init__给这个实例设置一些参数。

 

**python中主要存在四种命名方式：**

1、object #公用方法

2、_object #半保护

​         \#被看作是“protect”，意思是只有类对象和子类对象自己能访问到这些变量，

​         在模块或类外不可以使用，不能用’from module import *’导入。

​        \#__object 是为了避免与子类的方法名称冲突， 对于该标识符描述的方法，父

​         类的方法不能轻易地被子类的方法覆盖，他们的名字实际上是

​         _classname__methodname。

3、_ _ object #全私有，全保护

​            \#私有成员“private”，意思是只有类对象自己能访问，连子类对象也不能访

​             问到这个数据，不能用’from module import *’导入。

4、_ _ object_ _   #**内建方法，用户不要这样定义**

 

python里面%d表数字，%s表示字符串，%%表示一个%；

单引号内嵌套单引号需要转义字符/;单引号内嵌套双引号不需要嵌套；

双引号内嵌套双引号需要转义字符/；双引号内引用单引号不需要转义字符；

 

#### python的异常处理机制
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
在没有return 的情况下：try首先执行，如有异常跳到except，如果没有执行else，finally是一直要执行的

有return的情况下： 不管怎样，finally部分的代码是一定执行的，所以finally中有return的话，就按照finally 执行后的结果return 即使try部分，except，else 部分有return ，也是要先执行finally，finally没有return 就返回去return； 如果有就在finally 处return 了，不会回到原来的地方。所以结果可能和预计的不太一样

 

### **self解释**

reference：https://www.cnblogs.com/jessonluo/p/4717140.html

1. self代表是类的实例，而不是类本身，self.__class__ 才是类本身
2. self只是一个约定俗成的写法，本身就是一个参数，所以是可以更改的，比如写成this
3. self是不能不写的，比如有一个成员函数是Test类的成员函数是test，实例是a，解释器运行的时候是Test.test(a)把实例当成一个参数传入到self的位置，所以self是不能不写的。有一种情况可以不写，那就是类方法，只能通过Test.test()这种方式去调用。

 

 

### **杂七杂八：**

- str.endswith(suff, start, end) start 与 end为可选参数，默认为0和str的长度
- python2 math.floor(5.5) 返回5.0 python3 math。floor(5.5) 返回5
- python 实现地址连接 os.path.join(addr1, addr2)