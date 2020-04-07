---
layer: post
title: 记账系统设计
category: Life
tags: 记账
description: 记账系统设计和代码流程
---

## 初衷

目前主流的记账软件基本都是手动录入，有的时候一些小额的收益支出懒得去详细的记录，而且支出的来源往往很多，支付宝、微信、银行卡、现金、信用卡等等，全部手动录入着实有点麻烦。纯手动记账的话，市面上任何一个记账软件都能满足要求，如果要求一定程度上自动记账， 且严重依赖支付宝的话，支付宝记账单功能就已经够用了。可是如果需要满足多来源自动/半自动记账，目前市场上相关软件很少，且通常需要提供支付宝，微信的授权，实在有点担心。于是就打算自己写一个记账系统，满足日常记账的功能即可。



## 设计思路和要求

### 功能

2. 显示交互界面
2. 显示账户总额，分渠道显示渠道总额
3. 增改查删，支出收益记录
4. 记录股票买入和卖出记录，在买入时记录买入成本，卖出时核算最终收益，入账
5. 支付宝，微信账单批量导入

## 实现思路

1. 由于记账工具主要为个人使用，实用性要求比美观性更强，使用tkinter包简单实现就好，如果后续有美观的要求再进行优化也不迟。
   ----------------分割线---------------------

   在开发过程中，发现tkinter的界面实在有点简陋，不能忍，果断选择pyqt

2. 分为账户总额，支付宝总额，微信总额，股票证券总额，银行卡总额，现金总额，其他总额



## SQL Server 安装

reference：

[SQL Server 安装教程最全](https://blog.csdn.net/NBbz2018/article/details/92669721)

[手把手教你使用python连接sql server 2014 本地数据库](http://www.pianshen.com/article/9608108141/)

第一个ref 基本就把安装的流程和可能遇到的问题涵盖了，第二个ref介绍了连接了本地数据库的方法，照着来，基本就没有问题了。



## SQL Server 语法

sql server和hive， mysql等语法有些许区别，在此记录一下用到过得指令

```
# 在sql server中实现limit，使用top n
select top 1 * from table where conditions

# 插入数据 insert data
insert into table[(fea1,fea2,fea3)] values(val1,val2,val3)

# 更新数据
update tablename set col=1, col=2 where conditions
```



## SQL Server + Python 使用

reference： [pymssql模块使用指南](https://blog.csdn.net/lin_strong/article/details/82868160)

首先安装pymssql module 用于连接Microsoft SQL Server， pip install pymssql

使用流程:

1. 创建链接
2. 交互操作
3. 关闭链接connect

直接上代码讲解

```python
import pymssql

config_dict = {
host = 'host'
user = 'user'
password = 'password'
db = 'db'
}

# 创建链接; 
# as_dict=False 返回值为字典形式，否则为list，autocommit=False
connect = pymssql.connect(config_dict)

# 创建cursor实例，与数据库的交互通过cursor进行，一个connect在任何时候只有一个cursor对象处于查询状态
cursor = connect.cursor()

# 执行sql语句， 特别注意如果sql语句中有字符串的化 需要在字符串外加上‘引号’
cursor.execute('''
sql sentence
''')

# 如果上面的sql语句对数据本身进行了修改 需要调用commit保存
#也可以在建立connect的时候autocommit设为True
connect.commit()

# 有返回值的sql语句 需要用到fetch方法来获得结果
cursor.execute('''
select * from table
''')

# fetchone()返回一条tuple，可以用while row： cursor.fetchone() 来循环打印
#fetchall() 返回一个list of tuple，一个tuple的数据分别对应select的字段
# fetchmany(n) 返回多条数据
row = cursor.fetchone() 

cursor.close()
connect.close()




```



## GUI设计

### Tkinter包使用

Reference: [python 图形化界面设计](https://www.jianshu.com/p/91844c5bca78)

tkinter module python自带，无需安装。

```python
'''
设置位置的方法主要有.pack() .place()

.place()方法主要参数
relx, rely, relwidth, relheight 相对与根窗口的宽度高度，起始位置， 左上角为起点；取值为0-1
x=x, y=x, width=w, height=h 绝对像素值

.pack()
默认按布局先后以最小占用空间的方式自上而下的排列控件实例，并保持控件的最小尺寸

'''


from tkinter import *
from tkinter import ttk
# 创建根窗体实例
root = Tk()
# 窗口名称
root.title('name')
# 窗口大小
root.geometry('480x480')

lb = Label(root, text='xxx').pack()

# 输入框，常用方法有.get()返回数据框的输入值 .delete(start,end)清空起止index之间的字符
en = Entry(root)
en.place(relx=rx, rely=ry, relwidth=rw, relheight=rh, x=x, y=x, width=w, height=h)

# 按钮，最主要的参数是command，按下之后调用的函数
btn = Button(root, text='xxx', command=lambda: func())
btn.pack()

# 下拉框 在ttk中，需要额外import, textvarible 需要提前定义
var = Stringvar()
bb = ttk.Combobox(root, textvarible = var, values=['v1','v2','v3'])
# 设置默认值， 把values index为0的值设置成默认值
# 一个很尴尬的地方，如果在函数中创建控件，设置默认值不会显示，只有把控件实例返回，在函数体外设置默认值
bb.current(0)

# 弹窗
import tkinter.messagebox
tkinter.messagebox.showinfo('title','info')
tkinter.messagebox.showwarning('title','warning')
tkinter.messagebox.showerror('title','error')

# 子窗口, 后续控件只需要指定窗口为top即可
top = tkinter.Toplever()
top.title('child window')


# 主循环
root.mainloop()
```



