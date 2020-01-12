---
layout: post
title: 	SQL&NoSQL
category: Resource
tags: SQL
description: SQL常用语法
---

## **NoSQL vs SQL**

**NoSQL**

1. you can create document without having to first define its structure
2. each documents have its own structure
3. the syntax vary from database to database
4. you can add field as you go

**Scalability**

NoSQL: _horizontally scalable_;  can handle more traffic by sharing or add more servers

SQL: _vertically scalable_;  increase the load by increasing CPU SSD or RAM

_So ultimately NoSQL become more powerful than SQL_

**The structure**:

NoSQL: document based; _key value pairs_; graph database or wide-column store

SQL: table based; good for multirow transaction application




## **关联性表特征：**

1. 行包含实体的示例数据
2. 列包含实体的属性数据
3. 单元值存储单个值
4. 每列数据类型相同，每列唯一名称
5. 行列顺序任意
6. 任意两行不重复




## **SQL DML 和DDL** 
_数据操作语言_（DML）_数据定义语言_（DDL）
DML： 查询和更新语句
-  SELECT
-  UPDATE
-  DELETE
-  INSERT
DDL:  创建或删除表格
-  CREATE DATABASE
-  ALTER DATABASE
-  CREATE TABLE
-  ALTER TABLE
-  DROP TABLE
-  CREATE INDEX
-  DROP INDEX



## DataType/Constraint

**datatype**

- smallint
- int
- serial
- float
- real float
- date
- time
- timestamp
- interval

**constraint**

- check condition
- Not null
- unique
- reference(must exist in a columns in other table)
- AUTO_INCREMENT

 


## **Primary key 和 Foreign key：**

- 理想的primary key 是较短的数字且永远不变
- 但有时没有理想的主键，生成代理键，如propertyID 对用户来说没有任何意义，通常隐藏
- primary key 和 foreign key 的value set 必须相同
- 主键必须是唯一的，但是可以使复合主键

**referencing table**(child table) => table contains foreign key

**referenced table**(parent table ) => table contains primary key

**primary key: Not null , unique**




## **Normalization：**

- 将一个具有多主体的表分割为一组表，使得每个表只有一个主体
- 设计良好的表 每个决定因子必须是候选键，也就是不允许出现表中局部函数依赖； 函数依赖的意思就是通过一个或者若干复合属性可以决定其他所有属性
- 非结构良好的表应该被拆分




## **SQL执行顺序**

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

every table in relational database is an entity, a row is a specific instance of that type, columns represent the common properties shared by the instance of entity



## **视图**



CREATE VIEW viewName AS SELECT * FROM table_name WHERE condition; 

DROP VIEW（viewName）

- 隐藏行与列
- 显示计算结果
- 隐藏复杂的SQL语法
- 分层组织内置函数



## 索引 

索引可以理解成一个hash table， SQL SERVER 提供两种索引，clustered index 聚类索引，nonclustered index 非聚类索引。

聚类索引和非聚类索引的区别可以用字典很好的示例，聚类索引类似拼音排序，字典本身就是按照拼音的顺序排序的，所以查询一个知道拼音的字的时候，只需要翻看拼音首字母所在的区域，不需要翻看整个字典。非聚类索引类似于偏旁部首查询，有一个中间的hash table 每一个字对应一个在字典中的位置，然后根据位置拿到具体内容，查询一个字的速度很快，但是查询大块的连续值表现不如聚类索引。

**每个表只能有一个聚类索引**




## 4种join详解

join：两张表中都出现的key，

left outer join: 出现在左边表中，但是不在右边表中的key

right outer join:  出现在右表，但是没有出现在左表的key

left join: 只出现在左表中的，可能出现在右表中，也可能没有的key

right join: 只出现在右表中，可能在左表也可能不在左表的key

full join: 只要出现过的key都算

hive中没有left outer join，right outer join 语句实现left outer join, right outer join

```mysql
SELECT a.col , b.col FROM a LEFT JOIN b WHERE b.key IS NULL
SELECT a.col , b.col FROM a RIGHT JOIN b WHERE a.key IS NULL
```







## Syntax 
**SELECT 语句**

*the order of the rows in the database table is unknown and unpredictable , when using limit clause, always use ORDER BY clause to control the order of rows*

e.g.

1. SELECT A.col1， A.col2 , B.col1, B.col2 ... from A INNER JOIN B on A.premaryKey = B.foreignKey
2. SELECT A.col1 as c1, A.col2 as c2 from A join B on A.PK = B.FK [**where** A.col3 is NULL or A.col4 is NULL]
3. SELECT **DISTINCT** col1 from table_name
4. SELECT ***** from table_name **LIMIT n [ OFFSET m]** (skip m rows)
5. SELECT * FROM table_name **LIMIT m,n**  (m is offset, n is the number of rows)
6. SELECT col1 col2 from table_name **ORDER BY col3 ASC, col4 DESC**
7. SELECT customer_id, sum(amount) from payment **GROUP BY** customer_id
8. SELECT customer_id, sum(amount) from payment **GROUP BY** customer_id **HAVING** sum(amount) >200
9. SELECT XXX **INTO** new_table from old_table #数据备份




**Aggregate function**

count(*) 返回表所有rows 的个数

count（col_name）返回列值得个数，忽略null

HAVING clause 是配合aggregate function 使用的，用来指定聚合之后的结果满足的条件 




**Subquery**

SELECT c1, c2, c3 from table_name where rental_rate> (SELECT AVG(rental_rate) from film)




**IN/Between/LIKE 语句**

- value IN (v1, v2, v3...)
- value BETWEEN low and high （需要注意具体的数据库是否是闭区间或者开区间）
- where firstName LIKE "jen%" 
  (%match any sequence of characters, _ match any single characters；
  [charlist] 字符列中的任何一个单一字符, [!charlist] 不是字符列中的任何单一字符)
- used in WHERE clause
- NOT IN
- <> not equal




**UNION 语句/EXCEPT/INTERSECT**

1.  queries have the same number of columns
2. the corresponding columns in those queries must have the compatiable data types
3. remove duplicated rows, if not want , use **UNION ALL**
4. SELECT XXX FROM XXX UNION SELECT XXX FROM XXX
5. EXCEPT ： in first query not in second query
6. INTERSECT: only the query that are identical in both result sets are returned
7. EXCEPT ALL , INTERSECT ALL not remove duplicated rows



**EXISTS/ANY/ALL 用法/WHERE 和 HAVING 的区别** 
_EXISTS/ANY/ALL 都是配合where 和 having 使用的_
EXISTS(subquery) 返回TRUE or FALSE subquery 返回至少一条数据为TRUE 否则为FALSE
EXISTS 表示subquery 只有有返回任何一条record，exists 为True, 否则为False
ANY 满足任何一个条件返回True，否则False
ALL 满足所有条件返回True，否则False

```
e.g.

#ALL 必须大于female 最大的age 才主查询才算是满足条件
select * from student where gender='male' and age > all(select age from student where gender='female')

# ANY 大于最小的female的age就算是满足条件
select * from student where gender='male' and age> any(select age from student where gender='female')

```

where 针对的是原始数据
Having 针对where 之后的数据，可以处理aggregate function



**SELF JOIN**

SELECT employ_name FROM employee AS e1, employee AS e2 where e1.employee_location = e2.employee_location AND e2.employee_name = "Joe"

or

SELECT e1.employ_name FROM employee AS e1,join employee AS e2 on e1.employee_location = e2.employee_location AND e2.employee_name = "Joe"






 **Case When 语句（SWITCH）**

CASE
	WHEN condition1 THEN value1
	WHEN condition2 THEN value2
	...
	WHEN conditionN THEN valueN

ELSE
	ELSE_value 
END

相当于其他语言中的switch语句，***如果满足第一个条件，则后面的判断不会执行***




**NULL**

- IFNULL( col, default_value)    -> MySQL
- COALESCE(col, default_value)  -> MySQL
- ISNULL(col, default_value) -> SQL Server

** Comments** 
Single line comments start with '--'
Multiline comments start with /\*and end with '\*/ '




#### TABLE RELATED
**UPDATE TABLE**

update table_name

set col1 = value1, col2 = value2... where condition



**INSERT**

INSERT into table_name (col1,col2,col3...)

values (v1,v2,v3...),(v1,v2,v3...)...
自己指定数据的方式太繁琐，可以通过select 语句批量导入数据
INSERT INTO table_name SELECT * FROM table_name




**CRATE TABLE**

- create table table_name [like another_table]
- create table table_name (col1 dataType constraint, col2 dataType constraint, table constraint, [inherits exist_table_name])
- create table table_name as select col1,col2 from table2 where conditions




 **DELETE**

- delete statement return number of rows deleted
- DELETE FROM table_name where condition [returning * (col1,col2,...)]
- DELETE FROM table



**DROP**

- DROP TABLE[ IF EXIST] table_name [cascade]  (drop dependence together)
- TRUNCATE TABLE table_name (delete records inside the table, not the table itself)




**Alter Table**

actions:

1. add remove rename columns
2. set default value for columns
3. add check constraint to a column
4. rename table
5. alter table table_name add column col_name dtype
6. alter table table_name drop column col_name
7. alter table table_name rename col_old_name to new_name
9. alter table table_name alter col_name type new_type



**Index**

- CREATE INDEX index_name ON table_name (col1, col2, ...)
- CREATE UNIQUE INDEX index_name ON table_name (col1, col2, ...)
- DROP INDEX index_name ON table_name;   --> MS access
- ALTER TABLE table_name DROP INDEX index_name;  --> MySQL




#### DATABASE RELATED

- CREATE DATABASE  databasename

- DROP DATABASE databasename

- BACKUP DATABASE TO DISK ='filepath'

- BACKUP DATABASE TO DISK ='filepath' WITH DIFFERENTIAL (only backs up the parts of the database that have changed since the last full backup)

  

#### **Advanced Usage** 

**COUNT conditional filtering**

```
SELECT COUNT( CASE WHEN fea >0 THEN fea ELSE NULL END) FROM TABLENAME 
# 统计fea >0 的条数
```





 


## **HIVE**

hive的语法与sql 有些许区别，记录一些遇到的有区别的指令

- ALTER TABLE name *CHANGE* column_name new_name new_type  # sql语句用的是alter

