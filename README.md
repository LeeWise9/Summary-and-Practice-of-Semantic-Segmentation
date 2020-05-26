# Summary-and-Practice-of-Semantic-Segmentation
This is a survey and practice of semantic segmentation network. Taking the satellite image segmentation as the research task, several classic neural networks based on deep learning are tested.

调研了目前较为常用的语义分割网络，并在一套卫星图像语义分割数据集上做训练和测试。代码基于 Keras 框架编写，网络结构根据实际情况做了一些调整。

数据集下载地址：https://pan.baidu.com/s/1i6oMukH （yqj2）

预训练模型：目前预训练模型暂不公开。

## SegNet

最为基础的语义分割网络之一，结构简单。

<p align="center">
	<img src="http://e0.ifengimg.com/06/2019/0311/BA9E7467F27A73CD0D720B0C4674A5BE28418C49_size60_w1000_h290.jpeg" alt="Sample"  width="500">
</p>

## UNet

同样是较为经典的语义分割网络，结构简单。

结构方面的细节推荐查看：https://www.cnblogs.com/fanhaha/p/7242758.html

<p align="center">
	<img src="https://camo.githubusercontent.com/f027d83c0fbd1076896498f5870cd2dfcb7757c6/68747470733a2f2f696d61676573323031372e636e626c6f67732e636f6d2f626c6f672f313039333330332f3230313830312f313039333330332d32303138303132323230303135383339372d313237353933353738392e706e67" alt="Sample"  width="500">
</p>

## FCN

全卷积神经网络是比较经典的分割网络。根据转置卷积上采样倍数的区别，FCN family 包括 FCN-8，FCN-16, 和 FCN-32。其中 FCN-8 的效果最佳。

<p align="center">
	<img src="https://img-blog.csdn.net/20160514051444532" alt="Sample"  width="500">
</p>


## ENet

设计了非对称的编码解码模块，旨在减少参数和计算量。

使用了可分离卷积、空洞卷积、跳跃连接等结构。

<p align="center">
	<img src="https://img2018.cnblogs.com/blog/1229928/201811/1229928-20181123200009509-2584933.png" alt="Sample"  width="500">
</p>


## DeepLab_v3

参考：https://www.jianshu.com/p/755b001bfe38

<p align="center">
	<img src="https://upload-images.jianshu.io/upload_images/4688102-a2569e23d72df245.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="Sample"  width="500">
</p>


## PSPNet

使用了金字塔池化和插值缩放方法。

参考：https://blog.csdn.net/u011974639/article/details/78985130

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20181220122525850.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE5NzQ2Mzk=,size_16,color_FFFFFF,t_70" alt="Sample"  width="500">
</p>


## ERFNet 

核心操作是 residual connections 和 factorized convolutions(空间可分离卷积)

参考：https://blog.csdn.net/baidu_27643275/article/details/98187098

<p align="center">
	<img src="https://img2020.cnblogs.com/blog/1467786/202005/1467786-20200523103802654-263289331.png" alt="Sample"  width="500">
</p>






















