---
layout: post
title: Hadoop 资源
category: Resource
tags: Hadoop
description: Hadoop 的一些架构知识，和常用指令
---



# **Hadoop Arichitecture**

**Name node**: keep track of where everything is

**Data node**: store data

process:

1. client node –>(ask where data is ) name node
2. name node –>(data address) client node
3. clicent node –>(fetch data by address) data node

如果 name node down掉怎么办？

1. backup metadata
2. secondary namenode
3. HDFS federation(each name node only manage specific namespace volume)

 

## 常用指令：

```
hadoop fs -mkdir  dir  创建文件夹
hadoop fs -ls path  显示路径下的文件
hadoop fs -du path 用字节显示大小
hadoop fs -du -h path 用最优单位显示大小
hadoop fs -rm -r 迭代删除文件夹下的所有文件
hadoop fs -cat path/to/file  查看文件
hadoop fs -mv srcpath targetpath   移动文件
hadoop fs -cp srcpath targetpath  复制文件
hadoop fs -touchz file 创建一个空文件
hadoop fs -tail file 输出最后1kb的内容
```

## 把hdfs上的文件复制到本地/本地文件上传到HDFS：

```
hadoop fs -get  hdfsPath localPath  成功返回0 失败返回-1 如果本地有同名文件，则失败 
hadoop fs -copyToLocal  hdfsPath localPath //在公司工程机上不支持，不知道是否是版本问题 hadoop fs -put  local/path/to/file   hdfs/path // 如果有同名文件，则失败
```

## 查看前几行，后几行,行数
```
 hadoop fs -cat /path/to/your/file | head -100
 hadoop fs -cat /path/to/your/file | tail -100
 hadoop fs -cat /path/to/your/file | wc -l
```
## 查看文件夹下的文件并按照文件大小/时间排序 
hadoop查看文件的时候，输出的第一个参数是权限，… 第5个是大小，6,7个是时间
```
hadoop fs -ls /path/to/your/dir | sort -k5   大小倒序排序，如果文件很多，不能显示全的话，推荐使用
hadoop fs -ls /path/to/your/dir | sort -r k5    大小正序排序 
hadoop fs -ls /path/to/your/dir | sort -r k6,7 时间倒序排序
hadoop fs -ls /path/to/your/dir | sort k6,7     时间正序排序 
```
