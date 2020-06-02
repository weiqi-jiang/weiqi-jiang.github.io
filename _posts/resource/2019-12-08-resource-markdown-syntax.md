---
layout: post
title: MarkDown
category: Resource
tags: MarkDown
description: markdown 常用语法
---

# <a id='index' >MarkDown 基本语法 </a>


## 斜体/粗体/ 删除线

```
_test_ / *test*  斜体
**test** 粗体
***test*** / **_test_** / _**test**_  粗体加斜体
~~test~~ 删除线
```
效果展示: 
1： *test* 
2： **test**
3： _**test**_ 
4： ~~test~~



## 标题
```
  # 一级标题（注意#和title之间有空值）
  ## 二级标题
  ### 3级标题
  ...
```

效果展示：

# 一级标题

## 二级标题

### 三级标题





## 代码块

```
  ```code here ``` 方便展示这里不换行，如果换行，就会有两个代码块
```





## 超链接/ 链接/ 锚点

```
1：行内式： [title](url attribution) 属性事鼠标悬停在链接上会显示的文字, url和attribution 之间有空格

2：参考式：
这是一个[测试网址][1]，用[测试网址][1]来进行[测试][2]

[1]:xxxx "测试地址"
[2]:xxx

3： 自动链接
<url> 自动把文字变成链接

4：锚点 貌似只支持标题 
 ## <a id='index'> xxx </a>
 跳转到[xxx](#index)
```

效果展示:

1: [test]('xxx' 'test')

2: 这是一个[测试网址][1]，用[测试网址][1]来进行[测试][2]

[1]:xxxx "测试地址"
[2]:xxx

3:  <'https://xxxx'>

4:  跳转到[主标题](#index)



**Reference** 

[简书： markdown语法手册]( https://www.jianshu.com/p/8c1b2b39deb0 )