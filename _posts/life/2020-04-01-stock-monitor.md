---
layout: post
title: A股每日监控报表
category: Stock
tags: Stock
description: 每日a股指标监控系统
---

## 出发点

A股一共有3000多只股票，如果每天人工去查看各种指标选股买入或卖出工作量太大，即使只关注沪深300成分股，工作量也很可观。于是萌生出写脚本来自动监控，定时发邮件，剔除掉大量无效股票之后，人工只需要排查满足特定条件的股票即可，工作量骤降。



## 模块

- 查询股票数据模块
- 生成报告模块
- 发送邮件模块
- 指标计算模块

整个流程是先查询沪深300成分股的最近日级K线，将结果喂给指标计算模块，计算出指标的值传给报告生成模块，生成word格式报告，保存到硬盘，最后发送邮件模块去同样的位置找到保存的报告发送邮件到指定邮箱。



### 查询股票数据模块

常用的股票数据来源有Wind，Tushare，Baostock，或者写爬虫爬取财经网站；

其中wind最准确，稳定，但是要付费

Tushare免费，需要注册，需要积分才能使用更高级的API，有股票期货外汇币圈等很多领域数据

Baostock免费，不需要注册，不需要积分，目前只提供股票数据

由于目前只需要用到股票数据，所以使用Baostock方便快捷，十分好用

详细用法见[文档]([http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5](http://baostock.com/baostock/index.php/首页)) 

Tips(坑)：

- 返回dataframe格式中数据均为string类型
- 交易额交易量可能太大，返回数据中使用科学计数法表示，所以需要eval(str)去转化



### 指标计算模块

目前实现了同花顺软件上的两个指标，主力吸筹和主力真吸货，用来判断主力是否在进行洗盘或是吸筹操作

主力吸筹函数定义

![img1](/assets/img/life/stock/zhulixichou.PNG)

主力真吸货函数定义

![img2](/assets/img/life/stock/zhulizhenxihuo.PNG)

实现过程中发现talib库和中国的交易软件上看到的结果不同，发现是函数的定义不同，参考[**深度教学帖——SMA与EMA：talib与中国股票行情软件的差异**](https://www.joinquant.com/view/community/detail/5f9e51671c7037744785776eea2e1d22 ) 

另外同花顺给出的公式中有几个方程存在除0的情况，导致结果是无限大，解决方法是把0值置换成一个很小的数例如0.0001



### 生成报告模块

import python docx模块，模块使用简单，但是功能有限，反正是个人使用，与其花里胡哨，不如简单粗暴

详细用法参见[文档](https://python-docx.readthedocs.io/en/latest/index.html) 

```python
from docx import Document
# 创建文档实例
document = Document()

# 0表示添加顶级标题段落，可以是0-9的整数，对应不同级别的标题
document.add_heading('title',0)

# 添加自然段
p = document.add_paragraph('text')
# 如果一个自然段中有文字需要加粗或者其他处理，需要在中间拆开
p.add_run('bold').bold=True
p.add_run('some text')
p.add_run('italic').italic = True

# 添加表格
table = document.add_table(rows=1, cols=2,style='Table Grid')

# 添加行,在表格最后	
row = table.add_row()

# 按照行和列访问表单元格，把单元格的内容设为xxx
cell = table.cell(0,1)
cell.text = 'xxx'

# 也可以通过下列方式访问单元格
row = table.rows
firstrow = table.rows[0]
firstrowcells = table.rows[0].cells
firstrowfirstcell = table.rows[0].cells[0]

# 添加分页符，从这里往后的内容会另起一页，即使现在这一页没有满
document.add_page_break()

# 添加图片
document.add_picture(filepath)

# 保存,可以改文件后缀名
document.save('document.docx')
```



### 发送邮件模块

主要用到的模块smtplib，email模块，其中smtplib负责和邮件服务器之间的通信，email负责邮件的编辑

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 根据不同的邮件客户端去改变，网易是126/163
mail_host = 'smtp.qq.com'
# 用户名 不带@
mail_user= '6173*****'
# 授权码，需要去qq邮箱设置，复制过来
mail_pass = 'qt**********'
# 发送方邮件地址
sender = '6173*******@qq.com'
# 接收方邮件地址
receiver = 'jia*******@163.com'

# 创建邮件对象，plain表示纯文本，utf-8表示编码
message = MIMEText('content here', 'plain', 'utf-8')
# 设置主题 发送方 接收方
message['Subject'] = 'title'
message['From'] = sender
message['To']=receiver

# 发送带附件的邮件
message = MIMEMultipart()
message['Subject'] = subject
message['From'] = sender
message['To'] = receiver

p1 = MIMEText(content, 'plain', 'utf-8')
p2 = MIMEApplication(open(file_path, 'rb').read())
p2.add_header('Content-Disposition', 'attachment', filename=filename)

message.attach(p1)
message.attach(p2)

# 发送邮件
try：
	smtpObj = smtplib.SMTP()
    # 25是端口号，默认就行,连接邮箱服务器
    smtpObj.connect(mail_host, 25)
    smptObj.login(mail_user, mail_pass)
    smptObj.sendmail(sender, receiver,message.as_string())
    smtpObj.quit()
except smtplib.SMTPException as e:
    print('error', e)
    
```

**Reference**

[简单三步，用python发邮件](https://zhuanlan.zhihu.com/p/24180606) 

## 目前进度以优化点

进度：

- 流程跑通，目前在本地主机上每天晚上7.30执行一遍

优化：

- 指标太少，需要更多的指标去辅助参考
- 在本地运行，下一步考虑如果任务量大，租一台服务器

