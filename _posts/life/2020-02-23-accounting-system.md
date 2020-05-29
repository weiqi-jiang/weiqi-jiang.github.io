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
5. 数据可视化
6. 支付宝，微信账单[自动]批量导入(考虑web 端爬虫实现)

## 实现思路



## 前置知识

### HTML(Hyper Text Markup Language)

html对于大小写不敏感

| element                                             | comment                                                      |
| --------------------------------------------------- | ------------------------------------------------------------ |
| \<html>                                             | 页面根元素                                                   |
| \<head>                                             | 在头部标签中可以插入\<title>, \<style>, \<meta>, \<link>, \<script>, \<noscript>, and \<base>. |
| \<base>                                             | 描述基本的链接地址，所有链接标签的默认链接                   |
| \<link>                                             | 定义了文档与外部资源之间的关系，通常用于连接css              |
| \<style>                                            | 定义css文件的引用地址，也可以直接添加样式                    |
| \<meta>                                             | 元数据，例如网页描述作者修改时间等                           |
| \<script>                                           | 用于添加脚本 通常是js                                        |
| \<title>                                            | 网页标题，浏览器最上方的那个标题                             |
| \<body>                                             | 可见的页面内容                                               |
| \<h1>"title"\</h1>                                  | 大标题                                                       |
| \<p>"paragraph"\</p>                                | 段落                                                         |
| \<a href='https://xxxx' target="_blank">"link"\</a> | 链接,href定义链接内容，target指定链接展示位置，本例为新窗口展示 |
| \<img src='path/to/img' width=xxx, height=xxx />    | img标签具有空内容，在开始标签中进行关闭                      |
| \<hr>                                               | 创建水平线，没有关闭标签                                     |
| \<!-- html comments -->                             | 注释                                                         |
| \<br>                                               | 在不添加p标签的情况在段内换行                                |
| \<b> “bold”\</b>                                    | 粗体                                                         |
| \<i> “italic” \</i>                                 | 斜体 更多文本格式化标签见reference                           |

一个常见的html结构

```html
<!-- 可视化的HTML 页面结构 -->
<!DOCTYPE html>
<html>
    <head>
        <title>"title here"</title>
    </head>
    <!-- 只有body部分是浏览器中显示的-->
    <body>
        <h1>"head1 here"</h1>
        <p>"paragraph here"</p>
    </body>
</html>


```

大多数标签都有属性，下面是一些通用的属性

| attr  | comment                               |
| ----- | ------------------------------------- |
| class | 定义一个或者多个类名，类格式从css引入 |
| id    | 定义元素唯一的id                      |
| style | 行内样式                              |
| title | 描述元素额外信息                      |

Reference

[HTML教程](https://www.runoob.com/html/html-tutorial.html)

### WSGI(Web Server Gateway Interface)

WSGI帮助我们开发人员专心生成HTML文档，不必关注HTTP请求，响应格式，只需要定义一个响应函数，就可以响应HTTP请求

遵循WSGI 规范的web后端系统有两个部分组成，wsgi web server, wsgi web application

web server主要负责高效的处理请求，可以是多线程，多进程； web application 负责处理业务逻辑。web server 接受到前端http请求后，调用web application接口处理请求，请求处理完结果返回给web server 然后返回给前端。

Reference

[WSGI接口]( https://www.liaoxuefeng.com/wiki/1016959663602400/1017805733037760)

[白月黑羽教python](http://www.python3.vip/doc/tutorial/django/02/)



## Django + Vue 环境搭建

### 创建Django 项目/APP

```
django-admin startproject projectname

'''
结构
|--manager.py
|--projectname
	|--__init__.py
	|--settting.py
	|--urls.py
	|--wsgi.py
'''

python manage.py startapp appname 
'''
结构
|--migration
	|--__init__.py
|--admin.py
|--models.py
|--tests.py
|--views.py
'''

```

### 创建Vue 项目

```
vue-init webpack frontend
```

要使Vue文件变成浏览器能解析的文件格式需要用到webpack

```
cd frontend 
# 根据package.json 安装依赖
npm install 
npm run build 

'''
npm run build 会生成一个dist文件夹
dist
|--index.html
|--static
	|--css
	|--fonts
	|--img
	|--js 
'''
```

### 连接Django和Vue

1. 在django项目中urls.py 指定主页路径

   ```python
   from django.contrib import admin
   from django.urls import path, include, re_path
   
   #添加
   from django.conf.urls.static import static
   from django.views.generic.base import TemplateView
   
   urlpatterns = [
       path('admin/', admin.site.urls),
       path('records/', include('recorder.urls')),
       re_path(r'^$', TemplateView.as_view(template_name='index.html')),
   
   ] + static('/', document_root='frontend/dist')
   ```

2. 在django项目setting.py文件中

   ```python
   TEMPLATES = [
       {
           'BACKEND': 'django.template.backends.django.DjangoTemplates',
           'DIRS': [os.path.join(BASE_DIR, 'frontend/dist')], # 添加
           'APP_DIRS': True,
           'OPTIONS': {
               'context_processors': [
                   'django.template.context_processors.debug',
                   'django.template.context_processors.request',
                   'django.contrib.auth.context_processors.auth',
                   'django.contrib.messages.context_processors.messages',
               ],
           },
       },
   ]
   
   
   # 添加
   STATICFILES_DIRS= [
   os.path.join(BASE_DIR, 'frontend/dist/static'),
   ]
   ```

### 运行django/vue项目

运行前端项目，主要是用来看前端效果，修改代码保存后，会自动更新效果

```
cd path/to/frontend_project
npm run dev
```

运行后端项目，如果前后端连接一切正常，看起来的效果和前端是一样的，如果不一样，可能是前端文件没有更新，需要"npm run build" 重新生成最新文件。

```
 python manage.py runserver
```



## 前端VUE





## 接口文档

### 列出所有消费记录

请求消息

```
get records/allrecords HTTP/1.1
```

响应消息

```
HTTP/1.1 200 OK
Content-Type: application/json
```

响应体

响应消息body中包含响应内容，数据结构为JSON格式

```
{
    "ret_code": 0,
    "records": [
        {
            "date":'2020-03-04',
            "id": 1,
            "amount": -100,
            "channel": "alipay"
        },
        
        {
            "date":'2020-03-04',
            "id": 2,
            "amount": 100,
            "channel": 'wepay'
        }
    ]              
}
```

### 添加消费记录

请求信息

```
post records/api HTTP/1.1
Content-Type:   application/json
```

请求体

```
{
    "action":"add_customer",
    "record":{
        "date":"2020-04-03",
        "amount": 10,
        "channel":"alipay"
    }
}
```

后端接受该请求后，在数据库中按照date amount channel字段调价消费记录

响应消息

```
HTTP/1.1 200 OK
Content-Type: application/json
```

响应体

```
{
    "ret_code": 0,
    "id" : 677
}
```

如果成功返回，ret_code为0， 返回失败为1， id为新记录的id号

```
{
    "ret_code": 1,    
    "error_msg": "输入格式错误"
}
```

### 修改消费记录

请求消息

```
PUT  records/api  HTTP/1.1
Content-Type:   application/json
```

请求体

```
{
    "action":"update_record",
    "id": 6,
    "newdata":{
        "date":"2020-03-04",
        "amount":10,
        "channel": 'alipay'
    }
}
```

响应消息

```
HTTP/1.1 200 OK
Content-Type: application/json
```

响应体

修改成功

```
{
    "ret_code": 0 
}
```

修改失败

```
{
    "ret_code": 1，
   'error_msg': '该id消费记录不存在'
}
```

### 删除记录

请求信息

```
DELETE  records/api  HTTP/1.1
Content-Type:   application/json
```

请求体

```
{
    "action":"del_record",
    "id": 6
}
```

响应消息

```
HTTP/1.1 200 OK
Content-Type: application/json
```

响应体

删除成功

```
{
    "ret_code": 0
}
```

删除失败

```
{
    "ret_code": 1,    
    "error_msg": "id为xxx的消费记录不存在"
}
```



## 前后端数据交互

### axios模块安装

```
npm install axios --save
```

### 导入axios模块

```
# main.js文件中
import axios from 'axios'
Vue.prototype.$axios = axios
```

### 使用示例

在组件中例如加上监听点击行为 @click.native='click_test'，注意一定要带.native否则不会触发，然后methods内加上对应方法 

```vue
<template>
<!-- ... -->
    
    <el-submenu index="1">
        <template slot='title'><i class='el-icon-more'></i>消费记录</template>
        <el-button @click.native="clickfun">添加记录</el-button>
    </el-submenu>
<!-- ... -->
</template>

<script>
export default {
  data () {
    return {
      tabledata: [
        {
          date: '2020-04-01',
          amount: 10,
          channel: 'xxx'
        }
      ]
    }
  },
  methods: {
	click_test() { 
        this.$axios.get('url') // 后台接口
        .then(response => {  // 请求成功
        	console.log('请求成功');
        	console.log(response.data);
        	this.course_ctx = response.dat
      })
		.catch(error => {  // 请求失败
      	console.log('请求失败');
      	console.log(error);
      })
  }
}
</script>
}
```

