---
layout: post
title: Data Visualization
category: Resource
tags: Data Visualization
description: 
---

## 常用包

### Folium

安装

```
pip install folium
```

简单使用

```python
# importation
from folium.plugins import HeatMap
import pandas as pd
import folium

# load data 
data = pd.read_csv('path/to/geo_data.csv')
'''
data 形如
longtitude latitude value
113.32 23.12 3
113.21 22.28 4
...
'''

# 设置地图起始的经纬度和缩放大小，0趋向整个地球，越大精度越高，tiles更换地图背景风格
m = folium.Map(location = [23,113], zoom_start=7, tiles='Stamen Terrain')

# 绘制热力图
HeatMap([ [[lat,long, value] for lat, long in zip(data["latitude"],data["longitude"],data['value'])] for i in range(1,5)] ).add_to(m)
```

![heatmap](C:\Users\Jiang\AppData\Roaming\Typora\typora-user-images\image-20200730191501456.png)

如果要绘制动态热力图，使用HeatMapWithTime, 但在使用过程中遇到只显示基本地图，没有数据点和没有时间变化的情况，参考stackoverflow也没有成功解决，具体情况见reference 4描述

**Reference**<br>[文档](https://python-visualization.github.io/folium/)<br>[知乎：python如何画出漂亮的地图？](https://www.zhihu.com/question/33783546)<br>[颜控+轻便——借助Folium实现动态热力图绘制](https://zhuanlan.zhihu.com/p/38193282)<br>[HeatMapWithTime can not work](https://github.com/python-visualization/folium/issues/1221)

### Pyecharts

安装

```
pip install pyecharts

# 地图包,形式json[{"area":[longitude,latitude]}]
pip install echarts-countries-pypkg 
pip install echarts-china-provinces-pypkg 
pip install echarts-china-cities-pypkg 
pip install echarts-china-counties-pypkg 
pip install echarts-china-misc-pypkg
pip install echarts-cities-pypkg
```

简单使用

```python
from pyecharts.charts import Geo
from pyecharts import options as opts
import pandas as pd 

# load data 
data = pd.read_csv('path/to/geo_data.csv')
'''
data 形如
longtitude latitude value
113.32 23.12 3
113.21 22.28 4
...
'''

# 地图上加上坐标点
for i in range(len(data)):
    (geo 
    .add_schema(maptype="world") # 设置地图为世界地图
    # 加入自定义的点，格式为
    .add_coordinate(data.iloc[i]['hotelid'],data.iloc[i]['longitude'],data.iloc[i]['latitude'])
    # 为自定义的点添加属性：点的名称，[(点的特征，点的颜色)]，点的大小
    .add("station", [(data.iloc[i]['hotelid'],80)],type_=10,)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))#设置系列配置项中的标签配置项
    .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(is_show=False), #设置全局配置项中的视觉映射配置项
            title_opts=opts.TitleOpts(title="test"), #设置全局配置项中的标题配置项
        )
    )
```



**优缺点**

1. 地图风格不方便修改，使用卫星图需要申请百度地图appkey



**Reference**<br>[文档](https://pyecharts.org/#/zh-cn/intro)<br>[【Pyecharts】全球疫情蔓延趋势图～](https://www.kesci.com/home/project/5eea2aa8e5f796002c2b53e7)<br>[Pyecharts 根据经纬度和量值的大小，画出散点图](https://blog.csdn.net/weixin_41666051/article/details/83245993)<br>[pyecharts项目相关数据集和访问接口](https://pyecharts.readthedocs.io/zh/latest/zh-cn/datasets/#_11)<br>[分享一次自己使用 pyecharts 模块 画地图，一路踩坑和填坑的经历](https://blog.csdn.net/weixin_41563274/article/details/82904106)<br>[217 countries&regions map](https://echarts-maps.github.io/echarts-countries-js/preview.html)

### Kepler.gl

to do

**Reference**<br>[文档](https://docs.kepler.gl/)<br>[github](https://github.com/keplergl/kepler.gl)<br>

