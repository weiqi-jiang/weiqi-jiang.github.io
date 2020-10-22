---
layout: post
title: 数据库相关技术
category: Tech
tags: database
description: 涉及数据库主流优化技术
---

## 概念&对比

**Database & Data Warehouse**： 传统的关系型数据库主要应用是**基本的事务处理**，数据经常更改，变动，**读写都有优化**，主流数据库有MySQL, SqlServer等；数据仓库**主要应用是OLAP**，支持复杂的分析操作，数据不经常变动，**一般是只读优化**，更偏向于只读操作, 主流数据仓库有Redshift, Hive等。

**Hadoop**: 分布式计算(MapReduce) + 分布式文件系统(HDFS), HDFS提供数据分布式存储，MapReduce提供分布式计算方案

**Hive**： 基于Hadoop的**数据仓库**，一部分功能可以理解成SQL解释器，hive把sql翻译为MapReduce，让开发人员写SQL语句来计算HDFS上的结构化数据，优势在于离线计算，实时性很差。

**Hbase**： Hadoop Database 的简称,k-v结构，NoSQL数据库，适用于海量明细数据的随机实时查询，hbase只是利用hdfs管理数据的持久化文件(HFile),**和MapReduce没有关系**,优势在于实时计算，所有实时的数据都存入hbase中，客户端通过API访问hbase，实现实时计算。

**Reference**<br>[知乎： HBase 和 Hive 的差别是什么，各自适用在什么场景中？](https://www.zhihu.com/question/21677041)<br>[大数据之hadoop / hive / hbase 的区别是什么？有什么应用场景？](https://juejin.im/post/6844903734699376648)

## Cube

cube基本应用是**为了实现OLAP**, cube可以理解为多维pivot table，每个维对应一个观察数据的角度，可以理解为sql中group by语句中的字段，每个单元存在聚合度量值，cube提供多维视图，允许预计算和快速访问聚合数据，每个维有一个表与之关联，称为**维表**。

![](/assets/img/tech/database/cube.png)

**什么时候需要cube？**数据量大且分析需求频繁，cube可作为sql和数据库数据的中间层，数据量庞大时，sql提取聚合数据耗时很大，提前按各维度聚合cube，之后的dice，pivot，drill-down，roll-up, slice操作的耗时会减少很多。

**Reference**<br>[BI cube的前世今生：商业智能BI为什么需要cube技术](http://blog.itpub.net/30056930/viewspace-2089339/)<br>[数据立方体(cube)](https://www.cnblogs.com/sthinker/p/5965271.html)

## Kylin

首先明确一下Kylin主要是用来干什么的？kylin是用来**快速进行大数据查询**的，为什么不用hive？hive主要应用也是OLAP啊。很简单，因为hive太慢了。

![](/assets/img/tech/database/kylin.png)

麒麟技术核心：**预加载**和**cube**；预加载好理解，主要说明一下cube，假设有四个维度ABCD，一共的组合方式是16种(零维1种，1维4种，2维6种，三维4种，四维1种)，当然kylin针对维度组合有优化，暂且认为共有16种，按照这些维度组合方式分别形成cube，将结果**存入HBase**，存入过程进行了预聚合。kylin**适用于聚合查询场景**。

**Reference**<br>[在轻松气氛中浅谈——Apache Kylin](https://www.jianshu.com/p/26c18e6a30c3)<br>[Kylin、Druid、ClickHouse核心技术对比](https://mp.weixin.qq.com/s/unGF2-D7_HhIK8qZu7Vz0Q)