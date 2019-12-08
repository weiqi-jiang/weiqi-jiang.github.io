---
layout: post
title: Github Page + jekyll 搭建博客
category: Tech
tags: Github Page
description: 记录github page 搭建博客流程以及遇到的问题
---



## 前期准备

1. 安装ruby  [ruby download](https://rubyinstaller.org/downloads/) 
2. 配置环境变量 terminal 中输入ruby -v ， gem -v 不报错说明安装成功
3. terminal 中输入ridk install 安装MSYS2，选项选3
4. gem install bundler
5. gem install jekyll
6. jekyll -v 检查是否安装成功
7. 切换到博客文件夹下， jekyll serve --watch ， terminal中显示了url，进入url可以在本地查看博客



## 快速建站（使用github page初始主题）

1. 新建一个repository
2. 进入setting，修改repos的名字成username.github.io (username 为github的用户名)
3. 进入setting,  下拉，可以看到github page 的选项
4.  source 选master branch
5. username.github.io 即为初始网站url
6. optional： 选theme，自定义domain name



## 自定义Domain Name

1. 购买域名并解析到GitHub page的ip（ip ping一下博客的url就知道了，本人使用腾讯云DNSPods，买并解析的域名，如果不想实名，可以考虑go daddy，许需要信用卡，价格也稍微贵一点）
2. 进入repos 的setting 中的github pages 设置
3. custom domain 中填写购买的域名，done

**坑**：

1. DNS解析需要一个生效时间，并不是填写好，DNS服务器就可以马上生效的，所以不要心急
2. **DNS停止解析之后，依然生效**，原因有两个，一是**DNS服务器有缓存**，二是**浏览器有缓存**，需要清空浏览器记录（被chrome 坑惨了）



## 使用主题建站

1.  [jekyll theme](http://jekyllthemes.org/) 找一个合适的主题
2. 按照作者readme.md 文件中的安装方式安装（一般作者都会写使用方法）
3. 在_post文件夹中使用markdown编写博文就可以了，推荐使用Typora写markdown