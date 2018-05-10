# RNN解读
转载自知乎 [何之源](https://www.zhihu.com/people/he-zhi-yuan-16/activities) 的文章

## 完全图解RNN、RNN变体、Seq2Seq、Attention机制
**1&nbsp;&nbsp;&nbsp;&nbsp;从单层网络谈起**<br>
在学习RNN之前，首先要了解一下最基本的单层网络，它的结构如图：<br>
![](https://pic4.zhimg.com/v2-da9ac1b5e3f91086fd06e6173fed1580_r.jpg)
输入是x，经过变换Wx+b和激活函数f得到输出y。<br>

**2&nbsp;&nbsp;&nbsp;&nbsp;经典的RNN结构**<br>
![](https://pic4.zhimg.com/v2-a5f8bc30bcc2d9eba7470810cb362850_r.jpg)
图示中记号的含义是：
* RNN引入了隐状态h（hidden state）的概念，h可以对序列形的数据提取特征，接着再转换为输出。
* 圆圈或方块表示的是向量。
* 一个箭头就表示对该向量做一次变换。如上图中h0和x1分别有一个箭头连接，就表示对h0和x1各做了一次变换。

h2的计算和h1类似。要注意的是，在计算时，__每一步使用的参数U、W、b都是一样的，也就是说每个步骤的参数都是共享的__，这是RNN的重要特点，一定要牢记。
![](https://pic1.zhimg.com/v2-74d7ac80ca83165092579932920d0ffe_r.jpg)
依次计算剩下来的（使用相同的参数U、W、b）：
![](https://pic1.zhimg.com/v2-bc9759f8c642208a0f8514ccd0260b31_r.jpg)
目前的RNN还没有输出，得到输出值的方法就是直接通过h进行计算：
![](https://pic1.zhimg.com/v2-9f3a921d0d5c1313afa58bd3ef53af48_r.jpg)
剩下的输出类似进行（使用和y1同样的参数V和c）：
![](https://pic2.zhimg.com/80/v2-629abbab0d5cc871db396f17e9c58631_hd.jpg)
