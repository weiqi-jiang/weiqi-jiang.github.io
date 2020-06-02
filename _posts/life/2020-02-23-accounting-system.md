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
6. 支付宝，微信账单[自动]批量导入(考虑web 端爬虫实现）
7. 先实现网页端，后考虑做成桌面应用

### 网页是怎么诞生的？

1. 用户在浏览器内输入网址url
2. 浏览器向DNS发送请求得到服务器IP地址
3. 向目标IP发送http请求
4. 经过3次握手，建立TCP连接
5. 服务器向浏览器传送HTML,CSS,JS文件
6. 浏览器根据收到的HTML文件构造DOM Tree
7. 读取css文件，把样式放到对应的节点处，构造带样式的DOM Tree
8. 把DOM tree 节点按照从上到下，从左到右的顺序压入文档流
9. 根据文档流输出安排各元素在页面上的位置
10. 渲染，展示内容

(DOM: Document Object Model 简而言之DOM的作用是把web页面和脚本程序语言联系起来）

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

**HTML通用属性**

| attr  | comment                               |
| ----- | ------------------------------------- |
| class | 定义一个或者多个类名，类格式从css引入 |
| id    | 定义元素唯一的id                      |
| style | 行内样式                              |
| title | 描述元素额外信息                      |

**HTML Common Events**

1. **onchange** : an HTML element has been changed
2. **onclick**: user clicks an HTML element 
3. **onmouseover**: user moves the mouse over an HTML element
4. **onkeydown**: the user pushes a keyboard key
5. **onload**: the browser has finished loading the page 

常见的events，需要查看完整的events，参见reference 第二条。

**Reference**

[HTML教程](https://www.runoob.com/html/html-tutorial.html)

[HTML EVENTS](https://www.w3schools.com/jsref/dom_obj_event.asp)

### JavaScript

W3CSchool JavaScript tutorial知识点的整理 ，原文中很多编程的基础知识，稍微浏览一遍即可，重点在于JavaScript特有语法。

reference: [javascript w3cschool](https://www.w3schools.com/js/js_intro.asp)
常识：

1. JavaScript同时接受单双引号
2. JavaScript是用来干什么的？总的来说，JavaScript是用来修改HTML的
3. JavaScript运行在pc,平板，手机等设备的浏览器上，无需下载

以下是reference introduction给出的5个基础JavaScript能做的事情

```html
<!-- 修改id是demo的html标签的内容-->
<button type="button" onclick='document.getElementById("demo").innerHTML = "Hello JavaScript!"'>Click Me!</button>
<!-- 替换掉id为myImage标签的src属性，相当于替换一张图片-->
<button onclick="document.getElementById('myImage').src='pic_bulbon.gif'">Turn on the light</button>
<!-- 修改字体-->
<button type="button" onclick="document.getElementById('demo').style.fontSize='35px'">Click Me!</button>
<!-- 将标签隐藏-->
<button type="button" onclick="document.getElementById('demo').style.display='none'">Click Me!</button>
<!-- 展示之前隐藏的标签-->
<button type="button" onclick="document.getElementById('demo').style.display='block'">Click Me!</button>
```

这是一种较为简便但是不太“规范”的写法，JavaScript函数应该写在\<script\>\</script\>标签中,该标签可以放在html的\<body\> 或\<head\>标签中，可以放置任意多个, 需要注意的是，把script元素放置在\<body\>标签的时候最好放在在末尾，因为body标签包含的内容是用户可以看到的内容，编译script是需要时间的，影响用户观看体验

```html
<html>
  <head>
    <script>
      function myFunction() {
        document.getElementById('demo').innerHTML =' new content'
      }
    </script>
  </head>
  <body>
    <p>here is the body</p>
    <button type='button' onclick="myFunction()"> click button </button>
  </body>
</html>
```

同时JavaScript代码也不一定非要写在html文件中，可以另外写在单独的.js文件中，需要调用的时候从\<script\>标签引入，这么做的好处很明显，可以复用脚本代码

```html
<script src='path/to/myscript.js'></script>
```

**输出**

主要有四种形式分别对应

1. 写入标签
2. 直接写入HTML，用于调试，如果在HTML文件load完之后写入则覆盖所有现有HTML
3. 写入alert box
4. 写入浏览器console， 相当于打日志

```html
<script>function testFunc() {
  document.getElmentById('demo').innerHTML('new HTML content');
  document.write('xxxx');
  window.alert('xxx');
  console.log();
  }
  </script>
```

**变量**

用var 声明但是不需要指定数据类型，和python一样根据实际传入的数据类型决定，可以先声明再赋值，也可以声明的同时赋值，可以同时声明多个变量 var v1 = 1, v2 = 2, v3 = 3; 有一点值得注意，如果v1之前有过有效值，然后重新声明var v1 但是不赋值，v1的值是不变的，还是原来的值

**一些‘特殊’的操作符**

=== 表示数据类型和数据值都相同，!== 表示数据类型或数据值不相同 

**数据类型**

1. **primitive data** 表示single data without additional properties and methods（包含string number， Boolean，undefined）
2. typeof v1 返回v1的数据类型
3. null 表示空，typeof null is an object not null
4. undefined 表示variables without a value， 数据类型是undefined
5. undefined 和 null数值相同但是数据类型不同
6. array, dict类型的数据类型是object，function的数据类型是function
7. string类型的长度用length获取，txt.length
8. var x = new String('John') 如果一个变量是用new关键词创建的，数据类型统一是object，var s1='xx' ; var s2 = new String('xx'); s1=== s2 返回是false
9. js 数字永远是64-bit float 型，支持直接写科学技术法 var a = 1e10
10. 如果不同的数据类型之间进行操作，遵循从左到右的原则，10+20+‘’a‘’的结果是'30a',  'a' + 10 + 20的结果是‘a1020’; 如果是两个”numeric string“之间进行numeric operation，会自动把string转换成数字，例如‘100’ /'10' = 10
11. typeof NaN returns "number"
12. Infinity 和-Infinity 无限大，无限小， typeof Infinity returns "number"

**Object**

JavaScript 中object(或者说某个类的实例)通过dict来实现

```javascript
var person = {
    name: 'gimmy',
    age: 18
};
```

**Function**

JavaScript 函数结构 function funcName(p1, p2, p3) { ... }
js中**箭头函数**相当于匿名函数，类似python中的lambda表达式，用于简化函数定义，省略'{}' 和'return'，有时也不能省略，相当于map函数

箭头函数的详细用法参见[JavaScript初学者必看“箭头函数”与this](https://juejin.im/post/5aa1eb056fb9a028b77a66fd)

**Array**

1. array的element可以是object 也可以是其他任何类型

2. 正向访问array，指定从0开始的index即可，但是js不支持负数index，曲线救国就是array[array.length-1]来返回最后一个元素

3. 一点和python不一样的是，例如a = ['a', 'b', 'c', 'd'], index最大为3 ，但是js却可以a[6] = 'e'，python会报错out of index，但是js不会，index为4，5的元素用undefined填充
4. 当需要primitive data的时候 array会默认自动调用array.toString()方法，所以例如输出的时候，直接写array和array.toString()的结果是一样的

**Dates**

\# to be completed 未来用到再细看，目前先忽略

**Comparisons/Conditions**

      1. 支持三元表达式 var a = (age>18)? 'adult': 'young'
      2. 逻辑与或非采用&& || ！ 不知道and or not
      3. 支持switch语句，switch判断采用的是严格相等===， 必须值和数据类型都match才算满足case

**Loop** 

| methods                     | comments                                                     |
| --------------------------- | ------------------------------------------------------------ |
| for (i=0 ;i<5; i++){ ... }  | 标准格式                                                     |
| for (x in arr) { person[x]} | For/in Loop 用于访问object,此时x表示object对象的各种property ： |
| for (x of arr) { ...}       | For of loop 用于访问iterable object，x即元素值               |
| while (conditions){...}     |                                                              |
| do { ...} while (condition) |                                                              |

**Error**

try catch finally结构，可以用throw 声明手动抛出异常

```
try {
  if(condition) throw 'err';
}
catch(err){
  console.log(err)
}
finally{
  ...
}
```

**常用build-in  methods**

**String**

| methods                              | comments                                                     |
| ------------------------------------ | ------------------------------------------------------------ |
| str.lastIndexOf('x’ [, startindex])  | 返回最后一次出现的index 没有返回-1                           |
| str.indexOf('x' [, startindex])      | 返回第一次出现’x‘的index，如果没有返回-1                     |
| str.search('target string')          | 返回匹配的第一个index，不能指定startindex，但是可以匹配regular expression |
| str.slice(start, end)                | 切片，如果省略第二个index，则表示从start到string结尾，可以接受负数index |
| str.substring(start, end)            | 和slice一样，但是不能接受负数index                           |
| str.substr(start, length)            | 指定起始index和长度，省略length，则返回剩余全部，接受负数index |
| str.replace('oldstring','newstring') | 使用newstring替换oldstring                                   |
| str.toUpperCase()                    |                                                              |
| str.toLowerCase()                    |                                                              |
| str0.concat(str1, str2,...)          | 连接一个或多个str                                            |
| str.trim()                           | 去掉两端的空格                                               |
| str.charAt(position)                 | 返回position处的字符，如果没有返回空str                      |
| str.charCodeAt(position)             | 返回unicode                                                  |
| str.split()                          | 与python split方法一样                                       |

**Numbers**

| methods                                             | comments                                                |
| --------------------------------------------------- | ------------------------------------------------------- |
| num.isNaN()                                         | 判断是否是数据(Not a Num)                               |
| num.toString(base)                                  | 输出以base为进制数的number                              |
| num.toString()                                      | 不加base 就是输出string类型                             |
| num.toExponential(n)                                | 输出科学计数法表示的数，字符串类型，n表示小数点后的数位 |
| Number(x)                                           | 强制类型转化，如果不能转化成number，返回NaN             |
| Number.MAX_VALUE,Number.MIN_VALUE                   | JS中的最大最小number                                    |
| Number.POSITIVE_INFINITY; Number.NEGATIVE_INIFINITY | 无限大无限小，return on overflow                        |

**Array**

| methods                                                      | comments                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| arr.length                                                   | arr长度                                                      |
| arr.forEach( myfunction)  function myfunction(value [, index, arr]) {...} | 对每一个array中的元素调用function，value 表示元素值，index为元素索引，arr为array本身。如果用不到后面两个参数可以省略function内部对元素的修改不会影响到array本身 |
| arr.map(myfunction) function myfunction(value [,index, arr]){...} | 返回一个新arr不修改原来的值                                  |
| arr.filter(myfunciton) function myfunction(value[, index, arr]){return value > 10} | 保留满足myfunction中指定的条件的元素                         |
| arr.push(x)                                                  | 返回new array的length                                        |
| arr.pop()                                                    | 弹出最后一个元素并返回                                       |
| arr.shift()                                                  | 弹出第一个元素并返回                                         |
| arr.unshift(x)                                               | 把x插在array的头部                                           |
| arr.toString()                                               | 用逗号拼接列表元素返回一个string                             |
| arr.join('separator')                                        | 用separator连接列表元素返回一个string， 和(‘separator’).join(List)在python中一样 |
| delete arr[index]                                            | 把index处的元素置为undefined,位置保留，array长度不变，推荐使用pop和shift代替 |
| arr.splice(index, removenum, new element1, new element2...)  | index表示从哪里开始插入，闭区间; removenum表示从index处需要删除掉的元素数，然后把后面的若干个new elements 插入进去； 可以用来删除指定index处的元素而不留下undefined 元素 |
| arr.concat(x)                                                | x可以是另**若干个**array，可以是若干个string                 |
| arr.slice(startindex [, endindex])                           | 从startindex 到 endindex 开区间，“切”出一个新的array，不会改变原来的array， 如果缺省endindex，且切到array的最后一个元素 |
| arr.sort()                                                   | sort an array alphabetically, 如果sort number，会把number当成string类型对待，导致错误的结果'20'>'100' |
| arr.sort(function(a,b) {return a- b})                        | 自定义“大小” 返回值为负数，表示a小于b，大于0表示a大于b 等于0相等 |
| arr.reverse()                                                | 反转array中的元素顺序，可以先sort然后再reverse 达到倒序sort的目的 |
| Math.max.apply(null, arr)； Math.min.apply(null, arr)        | Math 模块直接找最大最小值                                    |
| arr.reduce(myfunction) function myfunction(init, value [, index, arr]){return init+=value} | 计算“范数”，从左到右如果不指定init，为空，如果指定则arr.reduce(myfunciton , 100) |
| arr.reduceRight(myfunction)                                  | 从右到左                                                     |
| arr.every(myfunciton)                                        | 判断是否所有元素都满足myfunction指定的条件，如果是返回true，否则false |
| arr.some(myfunciton)                                         | 判断是否有一些元素都满足myfunction指定的条件，如果是返回true，否则false |
| arr.indexOf(value [,startindex])                             | 和str.indexOf一样                                            |
| arr.lastIndexOf(value [,startindex])                         | 和str.lastIndexOf一样                                        |
| arr.find(myfunction)                                         | 返回第一个满足myfunction条件的元素值                         |
| arr.findIndex(myfunction)                                    | 返回第一个满足myfunction条件的元素index                      |

**Class**

用static修饰的方法，只能通过类名.方法名 car.myfunc() 调用，不能用实例名.方法名 newcar.myfunc()调用

```
class car {
  constructor(brand){
    this.brand = brand
  }
  [static] myfunc(){
    ...
  }
}

newcar = new car('Ford');

```

### WSGI(Web Server Gateway Interface)

WSGI帮助我们开发人员专心生成HTML文档，不必关注HTTP请求，响应格式，只需要定义一个响应函数，就可以响应HTTP请求

遵循WSGI 规范的web后端系统有两个部分组成，wsgi web server, wsgi web application

web server主要负责高效的处理请求，可以是多线程，多进程； web application 负责处理业务逻辑。web server 接受到前端http请求后，调用web application接口处理请求，请求处理完结果返回给web server 然后返回给前端。

Reference

[WSGI接口]( https://www.liaoxuefeng.com/wiki/1016959663602400/1017805733037760)

[白月黑羽教python](http://www.python3.vip/doc/tutorial/django/02/)



## Django + Vue + Element-UI环境搭建

### Django/Vue/Element-UI安装



安装django： 参考[django 安装]( https://www.runoob.com/django/django-install.html)

安装Vue:

1. 首先下载[node.js](https://nodejs.org/zh-cn/download/)

2. ```
   npm -v 检查npm是否正确安装 
   npm install --global vue-cli
   ```

安装element：

```
npm i element-ui -S
```

**Reference**

[Vue框架Element UI 教程-安装环境搭建](https://www.jianshu.com/p/ab3c34a95128)

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

\# to be completed

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

### Vue中引入Element-UI

```python
# frontend main.js中
// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'

#-- 添加内容
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
Vue.use(ElementUI)
#--

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
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

### 前后端调试

后端运行python指令运行服务，方便前后端联调的话，最好指定host和port，不使用默认host和port

```
# 一般后端接口使用80
python manage.py runserver localhost:80 
```

这只是跑起来了后端服务，具体的逻辑怎么测试呢，前端给一个请求，后端能不能按照逻辑给出响应呢，简单的方法就是自己“造”一个请求，利用python 的requests库

```python
import  requests,pprint

# 构建添加 客户信息的 消息体，是json格式
body = {
    "action":"add_record",
    "record":{
        "date":"2020-xx-xx",
        "amount": 100,
        "channel":"xxx"
        }
    }

# 测试消费记录列表接口
response = requests.get('http://localhost:80/record/api/?action=show_record')

# 测试添加记录接口
#response = requests.post('http://localhost:80/records/api/?action=add_record', json=body)

pprint.pprint(response.text)
```

前端vue项目可以在frontend/config/index.js中指定npm run dev的时候默认的port和host

```
// Various Dev Server settings
host: 'localhost', // can be overwritten by process.env.HOST
port: 8080, // can be overwritten by process.env.PORT, if port is in use, a free one will be determined
autoOpenBrowser: false,
errorOverlay: true,
notifyOnErrors: true,
poll: false
```

 console.log   debug  info 

\# to be completed

### 前后端数据交互

axios模块安装

```
npm install axios --save
```

导入axios模块

```
# main.js文件中
import axios from 'axios'
Vue.prototype.$axios = axios
```

使用示例

在组件中例如加上监听点击行为 @click.native='AddRecordFunc'，注意一定要带.native否则不会触发，然后methods内加上对应方法 

```vue
<template>
<!-- ... -->
	<el-button @click.native="AddRecordFunc">添加记录</el-button>
<!-- ... -->
</template>

<script>
import axios from 'axios'
export default {
  data () {
    return {
      form: {
        amount: '',
        channel: '',
        category: '',
        description: '',
        date: ''
      }
    }
  },
  methods: {
    // 添加记录函数
    AddRecordFunc () {
      var data = {
        'action': 'add_record',
        'record': {
          'date': this.form.date,
          'amount': this.form.amount,
          'channel': this.form.channel,
          'category': this.form.category,
          'description': this.form.description
        }
      }
      axios.post(`${this.baseURL}/`, data)
        .then((response) => {
          console.log(response)
        })
        .catch(function (error) {
          console.log(error)
        })
        .finally(function () {
          this.$message({message: '成功添加记录', type: 'success'})
          this.dialogVisible = false
        })
    },
  },
  created: function () {
    this.getrecord()
  }
}
</script>

```

## 前端VUE + Element-UI

.vue文件的注释

```vue
<template>
	<!-- html 注释-->
</template>

<script>
    // 单行注释
    /*
    多行注释
    */
</script>

<style>
    /*
    多行注释
    */
</style>
```

Vue项目下文件目录

```
|frontend
|--build
|--config
|--dist # 静态文件和index.html
	|-- static
		|--css
		|--fonts
		|js
	|--index.html
|--node_modules
|--src  # 最主要的一个文件夹
	|--assets  # 图片logo等资源
	|--components # 组件
	|--router
	|--App.vue  # 入口vue文件
	|--main.js
|--static
```

初始的app.vue文件引用了一个logo和component文件夹下的helloworld.vue组件，于是我们可以依葫芦画瓢在components文件夹下新建一个demo.vue文件，按照vue的语法规范写上一些页面元素，之后我们只需要在app.vue入口文件中引入这个组件即可,运行发现页面已经变成demo.vue文件中指定的样式

```vue
<template>
  <div id="app">
    <demo></demo>
  </div>
</template>

<script>
import demo from './components/demo.vue'
export default {
  components: {demo}
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
```

### 前端请求后端数据并在前端显示

首先要定义好前后端的数据接口格式，后端返回的数据字段和类型，然后在前端vue文件处理后端返回的数据即可，比如我们想要获得所有历史消费记录，展示在表格里：

```vue
<!-- vue文件中首先添加一个el-table元素 -->
<el-table :data='tabledata' style='width: 100%'>
    <el-table-column prop='date' label='日期' width='180'></el-table-column>
    <el-table-column prop='amount' label='金额' width='180'></el-table-column>
    <el-table-column prop='channel' label='渠道'></el-table-column>
</el-table>

```

然后在script部分修改为(最终顺利运行成果如下，中间有几个坑，会在后面细说)：

```vue
<script>
import axios from 'axios'
export default {
  data () {
    return {
      tabledata: []
    }
  },
  methods: {
    getrecord () {
      <!-- 这里对应第一个坑 -->
      axios.get(`${this.baseURL}/?action=show_record`)
        <!-- 这里对应第二第三个坑 -->
        .then((response) => {
          console.log(response)
          for (var i = 0; i < response.data.records.length; i++) {
            this.tabledata.push(response.data.records[i])
          }
        })
    }
  },
  created: function () {
    this.getrecord()
  }
}
</script>
```

**前端显示遇到的坑**

**第一个：** 前端请求跨域问题

reference： https://www.jb51.net/article/166134.htm # to be completed

**第二个：**后端返回response的结构是什么样的，需要怎么访问其中的元素？

最简单的解决办法是在前端代码对应位置上写上console.log(response)，然后打开浏览器，F12看log

**第三个：**Cannot set property 'tabledata' of undefined at eval

reference：https://blog.csdn.net/u011350541/article/details/80458708

其实就是箭头函数和this的用法， **箭头函数会默认绑定外层的this的值，不会使用自己this的值**，最外层的this就是window对象。 例如这里出错就在于不适用箭头函数的话，this指代getrecord对象，他是没有tabledata这个属性的，但是使用箭头函数的化，this指代window对象（网页打开的窗口对象）

```vue
<script>
import axios from 'axios'
export default {
  data () {
    return {
      tabledata: []
    }
  },
  methods: {
    clickfun () {
      axios.get(`${this.baseURL}/?action=show_record`)
        .then(function (response) {
          console.log(response)
        })
        .catch(function (error) {
          console.log(error)
        })
        .finally(function () {
          alert(1)
        })
    },
    getrecord () {
      axios.get(`${this.baseURL}/?action=show_record`)
      <!-- 需要在这里改为箭头函数 -->
        .then(function (response) {
          this.tabledata = []
          console.log(response)
          for (var i = 0; i < response.data.records.length; i++) {
            this.tabledata.push(response.records[i])
          }
        })
    }
  },
  created: function () {
    this.getrecord()
  }
}
</script>
```

**Reference**

[vue通过条件获取后台对应数据显示在表格](https://blog.csdn.net/huanxianxianshi/article/details/90479297)



## 后端Django

### HTTP请求URL路由

总的来说，app中的view.py 文件用来实现具体的响应函数，并在根文件夹的urls.py文件中调价路由记录

view.py 响应函数必须加上response参数

```python
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

import json

def showrecord(request):
    pass 

def addrecord(request):
    pass 

def delrecord(request):
    pass 

def updaterecord(request):
    pass

def dispatcher(request):
    if request.method == 'GET':
        request.params = request.GET
    elif request.method in ["POST", "PUT", "DELETE"]:
        # 此处遇到一bug，对应debug部分bug1
        request.params = json.loads(request.body)

    action = request.params['action']
    if action == 'show_record':
        return showrecord(request)
    elif action == 'update_record':
        return updaterecord(request)
    elif action == 'del_record':
        return delrecord(request)
    elif action == 'add_record':
        return addrecord(request)
    else:
        return JsonResponse({'ret_code': 1, "error_msg": 'error'})
```

urls.py

```python
#添加对应的代码之后，前端发回的url请求以record/api开始，讲给dispatcher处理

from django.contrib import admin
from django.urls import path

# view.py中的响应函数
from backend.views import dispatcher

urlpatterns = [
    path('admin/', admin.site.urls),
    #添加路由记录
    path('record/api/', dispatcher)
]
```

如果有很多url条目的话全部写在根目录下的urls.py 的话，该文件很大，而且也不方便管理，可以使用路由子表来把一类url聚在一起方便管理。

```python
#如果url以record开头，所有路由交给backend app中的urls.py 实现子路由(初始化不会创建urls.py文件，需要自己创建)

from django.contrib import admin
from django.urls import path

from django.urls.import include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('record/', include('backend.urls'))
]
```

### 创建和操作数据库

\# to be completed ORM

1. 定义一张表，就是继承一个model类
2. 定义字段就是定义类属性
3. 类方法就是表数据处理方法，包含增删改查

新建一个common app用于创建一个公共table，初始app的model.py文件是空文件, 添加下述代码

```
from django.db import models

class Record(models.Model):
	# 日期
    date = models.CharField(max_length=200)
	# 金额
    amount = models.FloatField()
	# 渠道，例如alipay wepay
    channel = models.CharField(max_length=200)
    # 类别 例如购物
    category = models.CharField(max_length=200)
    # 描述
    description = models.CharField(max_length=200)
```

数据类型对应表参加[datatype table](https://docs.djangoproject.com/en/2.0/ref/models/fields/#model-field-types)

目前我们只是继承了一个model类，相当于创建了一个表但是django不知道，我们需要在配置文件中加上响应的配置

```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # ADD, CommonConfig 是common app下app.py文件的一个类
    'common.apps.CommonConfig'
]
```

好了，现在django知道我们有common这个app并且在common app下有一个record表需要实现, 运行下面的指令会生成一个0001_initial.py,的脚本，这个脚本就是对应的进行数据库操作的代码, migrate指令运行数据库操作的代码，实际完成数据库逻辑

```
python manage.py makemigrations common
python manage.py migrate
```

如果修改Model.py 表的定义，都需要重新makemigrations和migrate

### Debug

bug1: 'bytes' object has no attribute 'read'

描述： 

出现在添加消费记录按钮被点击后，后端不能正确解析前端的post 请求。

原因在于错误使用了json.load() 函数，应该使用json.loads(); **loads操作对象是字符串**，把符合json格式的字符串转化为dict格式，**load操作对象是文件流**，也就是open('xx.json', 'r')，除此之外两者一样

reference：[ERROR 程序出错，错误原因：'bytes' object has no attribute 'read'](



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


