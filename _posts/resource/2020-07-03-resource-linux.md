---
layout: post
title: Linux
category: Resource
tags: Linux
description: 
---

## yum

```
# 安装package，-y参数遇到选项自动选择yes
yum install packagename 
yum -y install packagename

# 输出所有安装过的package, | grep xxx 过滤包名是否包含xxx
yum list installed 
yum list installed | grep xxx

# 输出所有可安装的package，支持*匹配
yum list 
yum list xxx[*] 
```

## crontab

业务涉及到使用crontab自动调用shell脚本，shell脚本中先激活anaconda环境，再运行py脚本, 如果直接按照terminal中的写法source activate pythonenv 并不能成功激活, 需要指定activate路径。正确的shell脚本应该是如下写法

```shell
#! /bin/sh
source /root/anaconda3/bin/activate python36;
python /root/xxx.py;
```

同时在terminal中输入crontab -e 编辑脚本，crontab -l显示当前调度的脚本，编辑脚本如下

```
# 每分钟执行一次shell脚本，输出重定向到/root/log文件
* * * * * source /root/path_to_shell/xx.sh >> /root/log
```

使用tail -f  /var/log/cron 动态查看crontab执行记录。

## Shell

```
# 记录日志
cmd > logfile  # 以覆盖的方式把正确输出重定向到文件
cmd >> logfile # 以追加的方式把正确的输出重定向到文件
cmd > logfile 2>&1 # 以覆盖的方式把正确和错误的输出重定向到文件
cmd >> logfile 2>&1 # 以追加的方式把正确和错误的输出重定向到文件
cmd >> logfile1 2>>logfile2 把正确的输出到file1，错误的输出到file2

# 多命令执行顺序
cmd1 ; cmd2   #顺序执行，没有任何逻辑
cmd1 && cmd2  #cmd1正确执行，cmd2才执行
cmd1 || cmd2  # cmd1不正确执行，cmd2才执行
 
```

### source; bash; sh; . 的区别

**source a.sh** 在当前shell去读取执行a.sh, a.sh不需要有执行权限，在a.sh中设置的变量，改变的环境都会作用于当前的process，例如在a.sh中使用source语句激活conda环境，那么a.sh中的剩余代码执行的python环境为conda环境，不是仅仅局限于child process； **bash/sh a.sh** 打开一个subshell执行sh文件，不需要有执行权限，a.sh设置的变量也不会影响到父shell；**./a.sh** 需要有执行权限，可通过chmod +777赋予所有权限，且打开subshell执行。

**Reference**<br>[Linux之shell详解](https://www.cnblogs.com/wuwuyong/p/11868651.html)<br>[知乎：怎么用shell脚本激活conda虚拟环境？](https://www.zhihu.com/question/322406344)<br>[linux里source、sh、bash、./有什么区别](https://www.cnblogs.com/pcat/p/5467188.html)

