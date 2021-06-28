---
layout: post
title: Linux
category: Resource
tags: Linux
description: 
---

## yum

```shell
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

```shell
# 每分钟执行一次shell脚本，输出重定向到/root/log文件
* * * * * source /root/path_to_shell/xx.sh >> /root/log
```

使用tail -f  /var/log/cron 动态查看crontab执行记录。

## Shell

**source; bash; sh; . 的区别**

**source a.sh** 在当前shell去读取执行a.sh, a.sh不需要有执行权限，在a.sh中设置的变量，改变的环境都会作用于当前的process，例如在a.sh中使用source语句激活conda环境，那么a.sh中的剩余代码执行的python环境为conda环境，不是仅仅局限于child process； **bash/sh a.sh** 打开一个subshell执行sh文件，不需要有执行权限，a.sh设置的变量也不会影响到父shell；**./a.sh** 需要有执行权限，可通过chmod +777赋予所有权限，且打开subshell执行。

**pipeline**

```shell
# 管道是将两个或者多个程序或者进程链接到一起，把上一个指令的输出作为下一个命令的输入 ‘|’ 表示管道符
# syntax:  command1 | command2 [|comand n] command1必须有正确输出，同时command2能处理1的正确输出结果
# 管道和重定向的区别在于管道是命令和命令之间，重定向是命令和文件之间。

ls | head -5  # ls会对文件名排序，展示前5个
cat file | less # 一次只展示一个屏幕大小
ls | grep -v "a"  #反向选择，输出不包含字符串的内容
```

**redirection**

```shell
# 记录日志
# 文件描述符 0 标准输入，1标准输出， 2 标准错误输出文件
cmd > logfile  # 以覆盖的方式把正确输出重定向到文件
cmd >> logfile # 以追加的方式把正确的输出重定向到文件
cmd > logfile 2>&1 # 以覆盖的方式把正确和错误的输出重定向到文件
cmd >> logfile 2>&1 # 以追加的方式把正确和错误的输出重定向到文件
cmd >> logfile1 2>>logfile2 #把正确的输出到file1，错误的输出到file2
cmd 1>>logfile1 2>>logfile2 # 一般把正确输出结果的1省略，当然可以写，和上面等价
# 特别注意 文件描述符和">"不能有空格，例如 2>>logfile2  2和“>>”之间不能空格但是
# “>>”和logfile2可以加，保险起见，就谁都不加

# 多命令执行顺序
cmd1 ; cmd2   #顺序执行，没有任何逻辑
cmd1 && cmd2  #cmd1正确执行，cmd2才执行
cmd1 || cmd2  # cmd1不正确执行，cmd2才执行
 
```

**tar**

```shell
# 打包/解包
# tar [-主指令][-辅指令]
# 主指令三选一，不能同时出现
# -c 打包；-x解包；-t查看有哪些文件
# 辅助指令
# -z是否有gzip属性，对应后缀为.tar.gz/.tgz 
# -j 是否具有bzip2的属性，对应后缀为.tar.bz2
# v压缩解包过程显示文件
# f指定文件路径，如果是打包，指定生成的文件路径和名称，解包就是指定包在什么路径下
# -f后面一定是直接接文件名，不能加其他的参数了
# -p 使用源文件的原来属性
# --exclude file 压缩时不要把file打包

tar -zxvf /path/to/xxx.tgz # 解压具有gzip属性的.tgz后缀文件，过程显示具体文件名
```

**find**

```shell
# find path -option
find . -name "*.c" # 把当前目录和子目录中以.c为后缀的文件输出
find . -type f # 把当前目录和子目录中所有一般文件列出
find . -ctime -20 # 把最近20天内修改过的文件列出
find / # 表示系统根目录，也就是列出所有文件
```

**ls**

```shell
# -a 展示所有文件和目录 包含.开头的隐藏文件 
# -l 出名字外，展示权限，拥有者，文件大小，文件形态，文件大小以k为单位
# -r 反序展示
# -t 按照创建时间排序
# -hl 文件大小以k，m，g显示，具体看文件大小，自动选择最佳单位

ll  # ls -l 的别名 
```

**nohup**

默认情况下，进程在前台运行，占用shell，无法进行其他操作，在启动参数结尾加上"&" 表示在后台启动，叫做后台进程或者job，后台进程会随着shell的结束而结束, 守护进程脱离控制台，不会受关闭控制台影响, 普通进程使用“nohup” 和 “&”的组合，达到守护进程的部分效果

```shell
nohup ./exec_file &
```

nohup 不间断的运行命令，即使shell关闭也不影响， 默认日志会输出到当前目录下的nohup.out文件，当然可以使用重定向把日志输出到指定文件下.

**rz,sz**

```shell
# 在已经连接服务器的情况下，在linux shell中输入
rz # 传输window文件到linux服务器，会弹出文件选择窗口
sz file #会弹出路径选择窗口，把file保存到window上选择的路径上
```

**Reference**<br>[Linux之shell详解](https://www.cnblogs.com/wuwuyong/p/11868651.html)<br>[知乎：怎么用shell脚本激活conda虚拟环境？](https://www.zhihu.com/question/322406344)<br>[linux里source、sh、bash、./有什么区别](https://www.cnblogs.com/pcat/p/5467188.html)<br>[linux 守护进程与用&结尾的后台运行程序有什么区别](https://blog.csdn.net/pursuer211/article/details/78932394)<br>[nohup /dev/null 2>&1 含义详解](https://blog.csdn.net/u010889390/article/details/50575345)

### Shell Script

```shell
echo "hello world"   # 输出

# 变量
# 变量 等号两边不能有空格
var1="abc"
# 使用变量, 推荐都加花括号，实在不加也行，但是容易出错
echo ${var1}
# 只读变量,尝试修改会报错
var2="123"
readonly var2
#删除变量,不能删除只读变量
unset var2

#字符串
# 字符串可以用单引号，也可以双引号， 推荐全部使用双引号
# 单引号所有内容原样输出，不能使用变量不能转义， 双引号可以使用变量，可以转义
name="zhangsan"
s=" my name is \"${name}\" ""
# 字符串拼接，主要有三种方式
name="lisi"
score="100"
res=$name$score
res="${name} any string you like ${score}"
res=$name"any string you like"$score
# 字符串长度, 这个时候一定要加花括号
echo ${#name}
#字符串截取,从下标为1截取到2， 下标从0开始,不写结束就是从1开始到最后
echo ${name:1:2} 
echo ${name:1}

#数组
#使用空格分隔
array=(1 2 3 4)
#访问元素
${array[0]}
#获取所有元素
${array[@]}
#获取数组长度
${#array[@]}

#命令行参数
# $0 文件名 $1 表示第一个参数 $n表示第n个
echo $1
# $* $@都是引用所有参数，但是*可以理解为一个整体，@是把参数拆开，在for循环访问时，*只循环一次，@循环参数个数次

#表达式
# 运算符两边必须有空格，必须被``符号包括，必须加上expr,乘法要转义才能实现
echo `expr 1 + 2`
echo `expr 1 \* 2`
# 关系表达式, 暂时只知道用在if 判断，只支持数字，不支持字符串
# -eq -ne -lt -gt -ge -le  -o或运算  -a and运算 && ||也分别表达或和与运算
if [1 -eq 2]
then 
	echo "True"
else
	echo "False"
fi
# 字符串运算符
# -z 字符串长度是否为0， -n 长度是否不为0 = 相等 !=不等
[-z "a"]
[-n "abc"]
["a" = "b"]

# 文件属性检测符， 常用如下
[-d $file] # 是否是路径
[-e $file] # 是否存在
[-f $file] # 是否是普通文件
[-s $file] #文件是否为空
[-x $file] # 文件是否可执行

#获得当前时间 date和"+"加号必须有，date后一个空格，剩下按照自己想要的格式写
time=$(date "+%Y%m%d-%H%M%S") 



```



## docker

安装流程参见Ref.1

**Reference**<br>[centos8安装Docker](https://blog.csdn.net/l1028386804/article/details/105480007)<br>[理解docker： docker安装和基础用法](https://www.cnblogs.com/sammyliu/p/5875470.html)<br>[理解Docker（2）：Docker 镜像](https://www.cnblogs.com/sammyliu/p/5877964.html)