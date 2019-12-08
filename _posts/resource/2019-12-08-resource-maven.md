---
layout: post
title: Maven 资源
category: Resource
tags: maven
description: 实习期间第一次接触的maven，秉承着实用主义，只记录常用的指令即可
---

# Maven 到底是干啥的



参考文档：

http://www.th7.cn/Program/java/201503/407878.shtml

首先maven项目和普通的JAVA项目差不多，只是多了一些功能而已

用户通过在pom.xml中添加配置，maven自动下载相对应的jar包，如果下载的jar包里面包含的pom.xml文件中有指定其他的jar包，会一并下载下来

## 项目文件结构

src/main/java  //存放.java文件

src/main/resources //资源文件

src/test/java //测试类.java文件

src/test/resources //测试类的资源文件

target  // 项目输出目录

pom.xml （project object model）

 

## 常用cmd

**mvn clean package 清除之前的包重新打包**

**mvn clean 删除target 文件夹**

**mvn package 生成jar包**

mvn compile 编译源代码

**mvn clean install -e -U   // -e 输出详情 -U强制更新**