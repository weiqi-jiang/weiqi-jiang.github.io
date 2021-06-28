---
layout: post
title: C++
category: Resource
tags: c++
description: 
---

大概从一个普通的开发者开发过程中会涉及到的知识点的顺序来排布

## Base

一般稍微大型的C++项目都需要头文件和源文件的分割。开发者在头文件(.h)中写类的声明，函数原型和#define常数等，不写具体的实现, 模板如下，目的是防止重复编译

```c++
#ifndef XXX_H
#define XXX_H

// CODE HERE 

#endif
```

源文件主要写头文件中已经声明的那些函数的具体代码，开头必须#include实现的头文件

## Syntax

**map**

```c++
// 内部实现是红黑树，插入元素的时候按照key自动排序
#include <map>

// 初始化
std::map<std::string, uint32_t> map_;
// 八个迭代器 带c表示返回const_iterator不允许对值修改，如果map为空begin()==end()
map_.begin();
map_.end();
map_.rbegin();
map_.rend();
map_.cbegin();
map_.cend();
map_.crbegin();
map_.crend();
// 迭代器可以加减操作，方便遍历
for(std::map<std::string, uint32_t>::iterator iter=map_.begin();
    iter != map_.end(); ++iter){
    // do something
}
// find 返回一个指向元素的迭代器，如果没有找到返回指向map_.end()的迭代器
std::map<std::string, uint32_t>::iterator iter = map_.find("a");
// 删除
map_.erase(iter);

```

**Reference**<br>[c++ map常见用法说明](https://blog.csdn.net/shuzfan/article/details/53115922)

**std::string**

```cpp
std::string a = "abc";
("%s", a); // 会报错format '%s' expects argument of type 'char*' but argument 2 has type 'const string' 需要string转char*

// std::string 转char*有三种方式
char *p = a.data();

char *p = a.c_str();

char p[40];
a.copy(p,3,0);// 5表示复制字符的个数，0表示起始位置

//char* 转string的方法
char* p="Hello";
std::string str = p;

// string 和long,int 之间的转化
#include <stdlib.h>

std::string a = "123456789";
int64_t b = atol(a.c_str());

```

**struct**

用来定义一个包含多个不同类型成员或者函数的数据结构， 与class不同的是结构体中成员默认是public，class默认是private 

```cpp
struct Test {
    int i;
    bool b;
    double d;
}
// 实例化
Test test;
// 也可以在定义时就声明变量
struct Test2 {
    int a;
    int b;
    int c;
}t1;

// 使用成员访问符访问成员
test.i;

// 定义指向结构的指针
Test *struct_pointer = &test;
// 使用指针访问结构成员 必须使用-> 符号
struct_pointer->b;
```

## Data Type

这里主要记录一些“非常规”的数据类型, 有的时候会看到uint8_t, int64_t类似的数据类型，其实这些带后缀"_t"的数据类型都是通过typedef来定义的关键词的别名，在C99标准中定义了这些数据类型

```cpp
#ifndef __int8_t_defined   
# define __int8_t_defined   
typedef signed char             int8_t;     
typedef short int               int16_t;    
typedef int                     int32_t;    
# if __WORDSIZE == 64   
typedef long int                int64_t;    
# else   
__extension__    
typedef long long int           int64_t;    
# endif   
#endif   
typedef unsigned char           uint8_t;    
typedef unsigned short int      uint16_t;    
#ifndef __uint32_t_defined   
typedef unsigned int            uint32_t;    
# define __uint32_t_defined   
#endif   
#if __WORDSIZE == 64   
typedef unsigned long int       uint64_t;    
#else   
__extension__    
typedef unsigned long long int  uint64_t;    
#endif   

// 格式化输出
uint16_t %hu;
uint32_t %u;
uint64_t %llu;  
// 其中uint8比较特殊，实际上它是一个char，输出uint8变量其实是输出对应的字符，而不是数值
```

**Reference**<br>[浅析C语言之uint8_t / uint16_t / uint32_t /uint64_t](https://zhuanlan.zhihu.com/p/37422763)

## Build

build过程大概分三个部分，**1预处理(preprocessing),2 编译(compiling),3.链接(linking)** 在build过程中只有cpp文件会build，.h文件会在预处理阶段整合到cpp中，即把.h文件的内容嵌入到cpp文件的上方。

**预处理**<br>cpp文件首先预处理变为translation unit，**translation unit仍然是文本代码文件**，它是传入编译器的基本单元，把cpp和h整合到了一起，并且去掉不必要的空格 换行之类的。

**编译**<br>translation unit传入编译器（compiler）之后会被编译成obj文件（二进制）即：高级语言->汇编语言->机器语言（二进制）,一般而言，生成的都是obj，但如果你想写一个第三方库文件，那么编译器会对应的生成lib（静态链接库）或者dll（动态链接库）文件。每一个cpp生成一个translation unit，然后编译生成一个obj，所以**cpp与obj是一一对应的**，每一个cpp都会**独立**编译出来一个obj文件。

链接<br>如果不依赖动态链接库或静态链接库，链接就是把所有obj链接；如果还依赖外部库，链接还包括lib文件。这里不包括dll文件，因为dll文件是在运行时才链接进来（其实也不应该叫链接，总之就是运行时才会加载进来，不会在链接生成exe的时候进来）

**Reference**<br>[C/C++ Build 过程简介](https://www.jianshu.com/p/bb1c46fae0e9)

## GFlags

```c++
#include <gflags/gflags.h>
#include <iostream>

// syntax DEFINE_TYPE(name, value, comment) 定义初始值
DEFINE_bool(b, true, "bool type");
DEFINE_int32(i1, 1, "int32 type");
DEFINE_int64(i2, 2, "int64 type");
DEFINE_uint64(i3, 3 "uint64 type");
DEFINE_double(d, 1.0, "double type");
DEFINE_string(s, "hello", "string type");

// 声明之后在程序的其他地方可以通过 FLAGS_name的形式来访问
cout<< FLAGS_s << endl;

// 声明和使用不在同一文件，需要用DECLEAR_type(name)来引入，然后就可以FLAGS_name的方式访问
// 一般的，在.CC文件中定义在.h文件中DECLEAR,其他文件包含头文件就可以使用
DECLEAR_bool(b);
```

在terminal或者.sh脚本中可以修改Flags的值

```sh
./exce_file --v=0 \
			--b=false
```

或者直接写一个开关文件.gflags   .sh脚本去调开关文件

```sh
./exce_file --flagfile="path/to/gflags_file"
```

```
--v=0
--logtostderr=1
--i3=4
```

## GLog

**日志等级**

日志等级按照严重性递增的顺序INFO<<WARNING<<ERROR<<FATAL,其中FATAL级别的日志会在打印之后退出程序，默认下ERROR,FATAL的信息在输出log的同时会输出到stderr

```c++
#include <glog/logging.h>

int a = 1;
int b = 2;
// log级别
LOG(INFO) << "the value of a is: " << a;
LOG(ERROR);
LOG(WARNING);
LOG(FATAL);

// 条件打印
LOG_IF(INFO, a<b) << "logging message";

// 抽样打印
LOG_EVERY_N(INFO, 10) << "logging message";

// 前若干次打印
LOG_FIRST_N(INFO, 5) << "logging message";

// check的功能和assert类型，check条件不满足，提前终止程序
CHECK(condition);
CHECK_EQ(a, b);  // 相等
CHECK_NE(a, b);  // 不相等
CHECK_LE(a, b);  // less equal
CHECK_GE(a, b);  // greater equal
CHECK_GT(a, b);  // greater than
CHECK_LT(a, b);  // less than

// VLOG 级别越低，越容易打印，v=0的VLOG总会显示， v=10时，小于等于10的VLOG会显示，大于10的不会打印，默认是INFO级别
int v = 2;
VLOG(v) << "LOGGING MESSAGE";
VLOG_EVERY_N(v, 10) << "logging message"; // V=2 每隔10条打一条日志
VLOG_IF(v, condition) << "message";
VLOG_IF_EVERY_N(v, condition, 10) << "message";
```

如果安装了GFlags，在编译时，默认带几个flags来控制GLOG的输出行为

```sh
--logtostderr=1 #log输出到terminal而不是log文件，bool类型，一般为0
--log_dir=path # 指定输出log的文件地址，默认输出在当前路径下
--v=2 #VLOG(v) 等级
```

## Proto

**syntax**

```protobuf
syntax = "proto2"  // 默认proto2 必须是非注释的第一行

import "other.proto"  // 使用其他proto文件中定义好的类型
package xx.xx.xx  // 可选，避免重名

message Request {
	// 每个字段必须指定required(必须一次), optional(0到1次), repeated(0到无数次)的其中一个
	optional string query = 1;  // "1"用来在二进制格式中标记字段，一旦使用，不能改变
	optional int32 page = 2;  // [1,15]编号占用一个字节，[16,2047]占用两个字节，所以常用的元素倾向于标记为[1,15]
}

// 在一个.proto 文件下可以定义多个消息
message Response {
	optional int32 res_code = 1 [default = 10];  // 如果元素不存在，对应字段置默认值,如果没有指明默认值，每个类型的元素会有对应的初始默认值
	optional string res_message = 2;
}

// 如果新版本中删除了某个字段，后面的用户，可能会重复使用某个标记号，这时如果不小心使用了旧版本的.proto加载，会有很严重的问题，为了保证标记号不被重复使用
// 在新版本中应该指定reserved标记符
message Foo {
	reserved 2, 7, 9 to 11;
}
// 使用枚举值
message Foo2 {
	enum Gender {
		FEMALE = 0;
		MALE = 1;
	}
}

// 如果有很多optional 字段且他们中只能有一个有值，则使用oneof修饰符节约内存, 在oneof定义中不能使用required，repeated或optional
message MyMessage {
	oneof Identification {
		string name;
		int32 id;
	}
}
// oneof会自动清理其他字段的值，所以如果多次set，只会保留最后的一次的效果
MyMessage.set_name("A");
MyMessage.set_id(1) // 这时name的值已经被清掉

```

**基于Proto的RPC服务**

```protobuf
// 首先编写proto文件
package goya.rpc.echo;
option cc_generic_services = true;

message EchoRequest {
  optional string message = 1;
}

message EchoResponse {
  optional string message = 1;
}

service EchoServer {
  rpc Echo(EchoRequest) returns(EchoResponse);
}
```

编译proto文件生成.pb.h/ .pb.cc 文件, 在pb.h文件中声明了多个类，一个继承自::google::protobuf::Service的EchoServer抽象类，一个继承自EchoServer类的EchoServer_Stub类。如果是多线程的话，还有很多继承自EchoServer的其他具体类。 这两个类中**抽象类EchoServer在之后将作为Server端逻辑处理类(EchoServerImpl)的基类，逻辑处理类中实现具体的Echo函数**。同时EchoServer也将作为EchoServer_stub的基类。

```cpp
class EchoServer_Stub;

// 一个接口或者说抽象类
class EchoServer : public ::google::protobuf::Service {
 protected:
  // This class should be treated as an abstract interface.
  inline EchoServer() {};
  
 public:
  virtual ~EchoServer();
  typedef EchoServer_Stub Stub;
  // 获取当前service的属性descriptor
  static const ::google::protobuf::ServiceDescriptor* descriptor();
  // 与proto文件中对应的调用方法Echo
  virtual void Echo(::google::protobuf::RpcController* controller,
                       const ::goya::rpc::echo::EchoRequest* request,
                       ::goya::rpc::echo::EchoResponse* response,
                       ::google::protobuf::Closure* done);
  
  // implements Service ----------------------------------------------
  // 获取当前Service的descriptor
  const ::google::protobuf::ServiceDescriptor* GetDescriptor();
  
  // 实际调用方法，对于每一个远程调用方法
  void CallMethod(const ::google::protobuf::MethodDescriptor* method,
                  ::google::protobuf::RpcController* controller,
                  const ::google::protobuf::Message* request,
                  ::google::protobuf::Message* response,
                  ::google::protobuf::Closure* done);
  // 利用descriptor获得请求prototype
  const ::google::protobuf::Message& GetRequestPrototype(
    const ::google::protobuf::MethodDescriptor* method) const;
  // 利用descriptor获得响应prototype
  const ::google::protobuf::Message& GetResponsePrototype(
    const ::google::protobuf::MethodDescriptor* method) const;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(EchoServer);
};

// 基于抽象类，实现了一个Stub具体类，封装了RpcChannel
class EchoServer_Stub : public EchoServer {
 public:
  EchoServer_Stub(::google::protobuf::RpcChannel* channel);
  EchoServer_Stub(::google::protobuf::RpcChannel* channel,
                   ::google::protobuf::Service::ChannelOwnership ownership);
  ~EchoServer_Stub();

  inline ::google::protobuf::RpcChannel* channel() { return channel_; }

  // implements EchoServer ------------------------------------------
  void Echo(::google::protobuf::RpcController* controller,
                       const ::goya::rpc::echo::EchoRequest* request,
                       ::goya::rpc::echo::EchoResponse* response,
                       ::google::protobuf::Closure* done);
 private:
  // 客户端与服务端交互的通道
  ::google::protobuf::RpcChannel* channel_;
  bool owns_channel_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(EchoServer_Stub);
};
```

服务侧.h 文件，声明逻辑实现类 继承自EchoServer基类，声明Echo函数，具体实现在.cc文件或者在.h文件中都可以写

```cpp
// 继承EchoServer实现类EchoServerImpl，并填充实现逻辑即可
class EchoServerImpl : public goya::rpc::echo::EchoServer {
public:
  EchoServerImpl() {}
  virtual ~EchoServerImpl() {}

private:
  // controller 用于记录上下文，在服务端，可以通过controller设置当前调用是否成功
  // done 回调函数，执行完服务端逻辑后调用该函数
  virtual void Echo(google::protobuf::RpcController* controller,
                    const goya::rpc::echo::EchoRequest* request,
                    goya::rpc::echo::EchoResponse* response,
                    google::protobuf::Closure* done) 
  {
    std::cout << "server received client msg: " << request->message() << std::endl;
    response->set_message(
      "server say: received msg: ***" + request->message() + std::string("***"));
    done->Run();
  }
};
```

启动服务端

```cpp
int main(int argc, char* argv[]) 
{
  RpcServer rpc_server;

  goya::rpc::echo::EchoServer* echo_service = new EchoServerImpl();
  if (!rpc_server.RegisterService(echo_service, false)) {
    std::cout << "register service failed" << std::endl;
    return -1;
  }

  std::string server_addr("0.0.0.0:12321");
  if (!rpc_server.Start(server_addr)) {
    std::cout << "start server failed" << std::endl;
    return -1;
  }

  return 0;
}
```

编写Client端， 实例化EchoServer_Stub，负责把所有的调用指向RpcChannel

```cpp
#include <iostream>
#include "rpc_controller.h"
#include "rpc_channel.h"
#include "echo_service.pb.h"

using namespace goya::rpc;

void print_usage()
{
  std::cout << "Use:         echo_client ip port" << std::endl;
  std::cout << "for example: 127.0.0.1 12321" << std::endl;
}

int main(int argc, char* argv[]) 
{
  if (argc < 3) {
    print_usage();
    return -1;
  }
  
  echo::EchoRequest   request;
  echo::EchoResponse  response;
  request.set_message("hello tonull, from client");

  char* ip          = argv[1];
  char* port        = argv[2];
  std::string addr  = std::string(ip) + ":" + std::string(port);
  // 实例化channel
  RpcChannel    rpc_channel(addr);
  // 实例化EchoServer_Stub
  echo::EchoServer_Stub stub(&rpc_channel);
  // 实例化controller
  RpcController controller;
  // stub 调用Echo函数
  stub.Echo(&controller, &request, &response, nullptr);
  
  if (controller.Failed()) 
    std::cout << "request failed: %s" << controller.ErrorText().c_str();
  else
    std::cout << "resp: " << response.message() << std::endl;

  return 0;
}
```

整个调用流程如下图

![](/assets/img/resource/c++/rpc_pipeline.png)

**Reference**<br>[github:sample, echo](https://github.com/goyas/goya-rpc/blob/master/sample/echo/echo_server.cc)<br>[基于Protobuf的简单RPC框架实现原理](https://blog.csdn.net/u014630623/article/details/106049993)

**Options**

不改变文件声明的含义，不过会影响特定条件下的处理方式

```protobuf
option java_outer_classname = "" // 表明想生成java类的名称，没有指定就会按照.proto的文件名，驼峰规则命名
option java_package = "" // 表明生成java类所在的包 
// 以上如果不生成java代码，不起任何作用
```



**Reference**<br>[Protobuf2 语法指南](https://colobu.com/2015/01/07/Protobuf-language-guide/)<br>[Protobuf3 语法指南](https://colobu.com/2017/03/16/Protobuf3-language-guide/)<br>[Protocol Buffer -如何实现可扩展性和向后兼容性？](https://www.coder.work/article/6641582)<br>[Protobuf 保留字段](https://docs.microsoft.com/zh-cn/dotnet/architecture/grpc-for-wcf-developers/protobuf-reserved)

**结构嵌套**

.proto文件中结构可以嵌套，但出现了一个问题，外部如果想定义一个efg类型的变量，类型名应该怎么写，应该写成namespace::abc_efg variable, 用下划线连接。

```protobuf
package namespace
message abc {
	enum efg {
		e = 1;
		f = 2;
		g = 3;
	}
	optional a = 1;
	optional b = 2;
	optional c = 3;
}
```

**获取enum变量字符串名字**

```protobuf
enum gender {
	male = 0;
	female = 1;
}
```

```c++
// 详细方法描述都可以在对应.pb.h文件中找到
// 方法一： namespace::gender_Name(value)
// 如果gender嵌套在message类型MyMessage内，使用namespace::MyMessage_gender_Name()
// 也可以使用namespace::MyMessage::gender_Name(),这个函数其实就是调用上面一个函数
std::string name = gender_Name(gender::male);

// 方法二，使用descriptor
const google::protobuf::EnumDescriptor *descriptor= gender_descriptor();
std::string name = descriptor->FindValueByNumber(1)->name();
// 得到序号
int number = descriptor->FindValueByName("male")->number();
```

**Reference**<br>[获取protobuf enum变量的字符串名字](https://blog.csdn.net/tang05505622334/article/details/90438625)<br>[ProtocolBuffer-doc-descriptor](https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor#EnumDescriptor)<br>[protobuf 的enum与string转换](https://www.cnblogs.com/hustcpp/p/12300014.html)

## 代码规范

**空行**

- .proto, .conf文件的末尾都要空一行

- 任何地方的空行都最多一行

**空格**

- 注释和“//” 符号直接要有一个空格， 行末注释要有至少两个空格
- for () { } 结构 “()”和for和花括号之间都要一个空格

**缩进**

- .proto下级元素较上级元素缩进两个字符
- 每一行不能超过100个字符，换行时，如果上一行有参数的话，第二行要和第一行的参数起始位置对齐，如果没有参数，至少缩进4个单位

**其他**

- for循环中，如果不使用迭代变量之前的值的话，使用++i 而不是i++

## Debug

**BUG 1:**  the vtable symbol may be undefined because the calss is missing its key function<br>**Solution 1:**  通常来说是基类中有虚函数没有在派生类中没有实现，导致链接出错的问题。比如析构函数没有实现，但是BUILD文件中没有正确添加依赖，也可能会导致该问题

**BUG 2:** XXX doesn‘t name a type<br>**Solution 2:** namespace没有正确写的问题

**BUG 3:** enumeration value "XXXX" not handled in switch <br>**Solution 3:**  需要写default， 如果default在逻辑上没有必要写，就写default：break；

