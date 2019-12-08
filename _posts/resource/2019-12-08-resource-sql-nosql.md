---
layout: post
title: 	SQL 语法
category: Resource
tags: SQL
description: SQL常用语法
---

### **NoSQL vs SQL**

NoSQL

1. you can create document without having to first define its structure
2. each documents have its own structure
3. the syntax vary from database to database
4. you can add field as you go

scalability

NoSQL: horizontally scalable, can handle more traffic by sharing or add more servers

SQL: vertically scalable; increase the load by increasing CPU SSD or RAM

**So ultimately NoSQL become more powerful than SQL**

The structure:

NoSQL: document based; key value pairs; graph database or wide-column store

SQL: table based; good for multirow transaction application

### **关联性表特征：**

1. 行包含实体的示例数据
2. 列包含实体的属性数据
3. 单元值存储单个值
4. 每列数据相同，每列唯一名称
5. 行列顺序任意
6. 任意两行不重复

### **Primary key 和 foreign key：**

- 理想的primary key 是较短的数字且永远不变
- 但有时没有理想的主键，生成代理键，如propertyID 对用户来说没有任何意义，通常隐藏
- primary key 和 foreign key 的value set 必须相同
- 主键必须是唯一的，但是可以使复合主键

### **Normalization：**

- 将一个具有多主体的表分割为一组表，使得每个表只有一个主题
- 设计良好的表 每个决定因子必须是候选键，也就是不允许出现表中局部函数依赖； 函数依赖的意思就是通过一个或者若干复合属性可以决定其他所有属性
- 非结构良好的表应该被拆分

### **视图**

create view viewName as { subquery }

drop view（viewName）

- 隐藏行与列
- 显示计算结果
- 隐藏复杂的SQL语法
- 分层组织内置函数

### **SQL执行顺序**

(8)select

(9)distinct

(11)top num columns

(1)from left_table

(3)join right_table

(2)on PK=FK

(4)where condiction

(5)group by columns

(6)with <CUBE | ROLLUP>

(7)Having condition

(10)order by columns

**sql代码不区分大小写，不过推荐关键词用全大写**

**关于怎么去理解SQL，怎么去理解关联性数据库中table和row，columns的关系**

every table in relational database is an entity, a row is a specific instance of that type, columns represent the common properities shareed by the instance of entity

**aggregate function**

count(*) 返回表所有rows 的个数

count（col_name）返回列 值得个数，忽略null

### **SELECT 语句**

**the order of the rows in the database table is unknown and unpredictable , when using limit clause, always use ORDER BY clause to control the order of rows**

e.g.

1. SELECT A.col1， A.col2 , B.col1, B.col2 ... from A INNER JOIN B on A.premaryKey = B.foreignKey
2. SELECT A.col1 as c1, A.col2 as c2 from A join B on A.PK = B.FK [**where** A.col3 is NULL or A.col4 is NULL]
3. SELECT **DISTINCT** col1 from table_name
4. SELECT ***** from table_name **LIMIT n [ OFFSET m]** (skip m rows)
5. SELECT * FROM table_name **LIMIT m,n**  (m is offset, n is the number of rows)
6. SELECT col1 col2 from table_name **ORDER BY col3 ASC, col4 DESC**
7. SELECT customer_id, sum(amount) from payment **GROUP BY** customer_id
8. SELECT customer_id, sum(amount) from payment **GROUP BY** customer_id **HAVING** sum(amount) >200
9. SELECT XXX **INTO** new_table from old_table

 

### **Subquery**

SELECT c1,c2,c3 from table_name where rental_rate> (SELECT AVG(rental_rate) from film)

### **IN/Between/LIKE 语句**

- value IN (v1,v2,v3...)
- value BETWEEN low and high
- where firstName LIKE "jen%" (%match any sequence of characters, _ match any single characters)
- used in WHERE clause
- NOT IN

### **UNION 语句/EXCEPT/INTERSECT**

1.  queries have the same number of columns
2. the corresponding columns in those queries must have the compatiable data types
3. remove duplicated rows, if not want , use **UNION ALL**
4. SELECT XXX FROM XXX UNION SELECT XXX FROM XXX
5. EXCEPT ： in first query not in second query
6. INTERSECT: only hte query that are identical in both result sets are returned
7. EXCEPT ALL , INTERSECT ALL not remove duplicated rows

### **EXISTS 用法/where 和 having 的区别** 

EXISTS(subquery) 返回TRUE or FALSE subquery 返回至少一条数据为TRUE 否则为FALSE

 

where 针对的是原始数据

Having 针对where 之后的数据，可以处理aggregate function

### **self join**

SELECT e1.employ_name FROM employee AS e1, employee AS e2 where e1.employee_location = e2.employee_location AND e2.employee_name = "Joe"

or

SELECT e1.employ_name FROM employee AS e1,join employee AS e2 on e1.employee_location = e2.employee_location AND e2.employee_name = "Joe"

### **INSERT**

INSERT into table_name (col1,col2,col3...)

values (v1,v2,v3...),(v1,v2,v3...)...

### **UPDATE**

update table_name

set col1 = value1, col2 = value2 where condition

[returning id,url ,name,description]

### **Create table**

- create table table_name [like another_table]
- create table table_name (col1 dataType constraint, col2 dataType constraint, table constraint, [inherits exist_table_name])

 

dataType:  smallint, int, serial, float, real float,date,time,timestamp,interval

constraint:  check condition, Not null, unique, reference(must exist in a columns in other table)

 

referencing table(child table) => table contains foreign key

referenced table(parent table ) => table contains primary key

**primary key: Not null , unique**

 

 

### **Delete**

- delte statement return number of rows deleted
- DELETE FROM table_name where condition [returning * (col1,col2,...)]

 

### **Alter Table**

actions:

1. add remove rename columns
2. set default value for columns
3. add check constraint to a column
4. rename table

 

- drop table[ if exist] table_name [cascade]  (drop dependence together)
- alter table table_name add column col_name dtype
- alter table table_name drop column col_name
- alter table table_name rename col_old_name to new_name
- alter table table_name alter col_name type new_type

 

 

# **HIVE**

hive的语法与sql 有些许区别，记录一些工作中用到的有区别的指令

### **Alter Table**

ALTER TABLE name RENAME TO new_name
ALTER TABLE name ADD COLUMNS (col_spec[, col_spec ...])
ALTER TABLE name DROP COLUMN column_name
ALTER TABLE name *CHANGE* column_name new_name new_type # sql语句用的是alter
ALTER TABLE name REPLACE COLUMNS (col_spec[, col_spec ...])

同时想删除掉一个字段的时候 用drop指令会出现

mismatched input 'column' expecting PARTITION near 'drop' in drop partition statement报错，目前没有解决

### **Case when 语句（SWITCH）**

**case when** condition1 **then** value1

**when** condition2 **then** value2

...

**when** conditionN **then** valueN

**else** else_value **end** as columns_name

相当于其他语言中的switch语句，***如果满足第一个条件，则后面的判断不会执行***