---
layout: pose
title: 二维码原理
category: Tech
tags: qr code
description: qr code principle
---

# 二维码(QR Code)

QR Code (quick response code)较之条形码能存储更多的信息，能表示更多的数据类型，一维条码只能在水平方向或者垂直方向上记录信息，而且只能由数字和字母组成，而二维码能存储汉字，数字，甚至图片等信息，它具有条码技术的一些共性：每种码制有其特定的字符集；每个字符占有一定的宽度；具有一定的校验功能等（[来自百科]([https://baike.baidu.com/item/%E4%BA%8C%E7%BB%B4%E7%A0%81/2385673?fr=aladdin](https://baike.baidu.com/item/二维码/2385673?fr=aladdin))）

实现原理是特定几何图形按编排规律在二维空间(所以被称为二维码)上分布，用黑白图形代表二进制的01，白色代表‘0’， 黑色代表‘1’，通过图像输入设备自动识读实现信息处理。二维码分为堆叠式二维码和矩阵式二维码，堆叠式二维码由多天短截的一维条码堆叠而成，矩阵式二维码以二维矩阵的形式，在对应点上用黑色表示1，空表示0.



# Python 简单实现

### qrcode 库

```python
​```
requirment: pip install qrcode
​```
import qrcode

# easy usage
img = qrcode.make('data here')
img.save('filepath/filename')

# advance usage
# 返回一个qrcode 对象
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_Q,
    box_size=10,
    border=4
				)
qr.add_data('data here')
qr.make(fit=True)
img = qr.make_image(fill_color='black', back_color='white')
img.save('filepath/filename')
​```
version: 1-40之间的整数， 控制qrcode的大小，1 最小表示21*21的matrix,（21+(n-1)*4） 设为None时，可以在make式make(fit=True)来自动调节
fill_color,back_color用来指定填充的和空白的部分的颜色
error_correction: 可选
    1. ERROR_CORRECT_L 大概7%的错误纠正率
    2. ERROR_CORRECT_M(default) 大概15%的错误纠正率
    3. ERROR_CORRECT_Q 25% 错误纠正率
    4. ERROR_CORRECT_H 30%错误
box_size =用来指定每一个box占据多少像素
border指定边缘多少个box宽度
​```


# 二维码中添加logo，如果logo过大，error_correction  需要调整参数
# 转换为RGB通道，否则插入的logo会被灰度化
img = img.convert('RGBA') 
qr_w, qr_h = img.size
icon = Image.open('logo_img_path')
icon_w, icon_h = icon.size
# 设置logo的长宽阈值，阈值根据error_correction 参数决定，这里是不能大于1/4
thres_w = int(qr_w/4) 
thres_h = int(qr_h/4)
if icon_w > thres_w:
    icon_w = thres_w
if icon_h > thres_h:
    icon_h = thres_h
icon = icon.resize((icon_w,icon_h), Image.ANTIALIAS)
icon = icon.convert('RGBA')
# icon 图片粘贴位置， target_w,target_h 分别是距离左上角的距离,可以只指定icon插入的左上角，也可以同时指定左上和右下角，如果图片大于这个大小，会被裁剪
target_w = int((qr_w-icon_w)/2)
target_h = int((qr_w-icon_w)/2)
img.paste(icon, (target_w, target_h)) # img.paste(icon,(target_w, target_h, target_w + icon.size[0], target_h+icon + size[1])) 
img.save('myqrcode.png')


```

### 效果：

![myqrcode](https://github.com/weiqi-jiang/weiqi-jiang.github.io/blob/master/pic/myqrcode.png?raw=true)

