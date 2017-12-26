# Churn Analysis SDK
## 简介

这是一个分析用户流失的程序，通过解析一个数据库形式的埋点数据，进行游戏用户流失分析，该环境运行在ubuntu16.04，python35，需要安装sklearn，pandas，numpy，XGBoost等包

为更加方便用户使用，现已经上述环境和代码封装于docker镜像中，并上传至阿里云镜像，名字为
<center>
**registry.cn-hangzhou.aliyuncs.com/jingchunzhen/churn_analysis**
</center>


## 输入输出

#### Step 1
用户需要在该项目中新建一个名为data的文件夹， 在data中存放需要处理的需要进行解析的数据库文件（sqlite），数据库文件中至少需要包含以下字段

<center>

字段|数据类型|意义
-----|-----------|---------
user_id|int|用户标识
op|text(string)|动作的名称
current_day|int|当前玩的天数
num_days_played|int|总共玩的天数
relative_timestamp|float|动作发生的时间

</center>

#### Step 2

```
python3 main.py -i XXX.db2 -d 1 -k t 
```
-i 之后的参数表示的是数据库文件的存放地址
-d 表示的是需要处理的是第i天的数据即首日用户流失分析，次留玩家流失分析，三留玩家流失分析
-k 参数t表示的是处理全部的动作信息，参数f表示仅处理由分类器提取出的较为重要的动作信息

#### 输出

输出为一个csv文件，存储在output文件夹中

<center>

字段|字段说明
----|---------
动作|埋点数据中动作的编码
留存比|点击该动作流失的人数与点击该动作未流失人数的比值
动作流失人数|点击该动作流失的人数
点击次数比|非流失玩家点击该动作的平均次数与流失玩家点击该动作的平均次数的比值
动作时段|该动作首次出现在游戏中的平均时间，以衡量该动作属于游戏前期，中期或者后期动作
随后平均时间间隔|卡点分析，判断该动作是否是一个潜在的卡点
随后中位数时间间隔|卡点分析

</center>

同时，还计算
* 动作留存比和动作随后时间间隔之间的Pearson系数，以验证留存和卡点之间的相关性，并绘制相关性曲线
* 动作平均卡顿时间
* 所有动作点击次数比的平均值，以验证某一个动作非流失玩家或者流失玩家的偏好性


## 项目结构

#### utils

- data_parser.py 处理原始数据，生成临时的数据，存储在temp文件夹下。
- data_analysis.py 统计一些数据的特征，如卡点分析，留存率和卡点之间的相关性系数等。

#### temp

存储生成的临时文件（包含处理数据之后的文件）

- clean.sh 清理临时文件夹

#### xgb
 
- xgb_model.py 进行流失用户分析的算法

#### main.py 

存放模型的流程

#### output

该文件夹存储最终生成的数据报表文件和曲线图

## Docker镜像使用说明

#### Step 1

安装docker

#### Step 2

```
docker pull registry.cn-hangzhou.aliyuncs.com/jingchunzhen/churn_analysis

docker run -t -i --name temp registry.cn-hangzhou.aliyuncs.com/jingchunzhen/churn_analysis /bin/bash

docker cp 宿主机中埋点数据的文件地址 temp: /home/workspace/data
```

#### Step 3

进入容器内

```
cd /home/workspace

bash run.sh 
```

#### Step 4

```
docker cp temp:/home/workspace/output/生成的数据报表文件 宿主机中的文件地址
```





