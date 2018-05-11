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

这就是最经典的RNN结构，我们像搭积木一样把它搭好了。它的输入是x1, x2, .....xn，输出为y1, y2, ...yn，也就是说，__输入和输出序列必须要是等长的__。<br>
由于这个限制的存在，经典RNN的适用范围比较小，但也有一些问题适合用经典的RNN结构建模，如：
* 计算视频中每一帧的分类标签。因为要对每一帧进行计算，因此输入和输出序列等长。
* 输入为字符，输出为下一个字符的概率。这就是著名的Char RNN（详细介绍请参考：[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)，Char RNN可以用来生成文章，诗歌，甚至是代码，非常有意思）。


**3&nbsp;&nbsp;&nbsp;&nbsp;N&nbsp;VS&nbsp;1**<br>
有的时候，我们要处理的问题输入是一个序列，输出是一个单独的值而不是序列，应该怎样建模呢？实际上，我们只在最后一个h上进行输出变换就可以了：
![](https://pic1.zhimg.com/80/v2-6caa75392fe47801e605d5e8f2d3a100_hd.jpg)

这种结构通常用来处理序列分类问题。如输入一段文字判别它所属的类别，输入一个句子判断其情感倾向，输入一段视频并判断它的类别等等。<br>


**4&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;VS&nbsp;N**<br>
输入不是序列而输出为序列的情况怎么处理？我们可以只在序列开始进行输入计算：
![](https://pic4.zhimg.com/80/v2-87ebd6a82e32e81657682ffa0ba084ee_hd.jpg)

还有一种结构是把输入信息X作为每个阶段的输入：
![](https://pic1.zhimg.com/80/v2-fe054c488bb3a9fbcdfad299b2294266_hd.jpg)

下图省略了一些X的圆圈，是一个等价表示：
![](https://pic2.zhimg.com/80/v2-16e626b6e99fb1d23c8a54536f7d28dc_hd.jpg)

这种1 VS N的结构可以处理的问题有：
* 从图像生成文字（image caption），此时输入的X就是图像的特征，而输出的y序列就是一段句子
* 从类别生成语音或音乐等


**5&nbsp;&nbsp;&nbsp;&nbsp;N&nbsp;VS&nbsp;M**<br>
下面我们来介绍RNN最重要的一个变种：N vs M。这种结构又叫Encoder-Decoder模型，也可以称之为Seq2Seq模型。<br>
原始的N vs N RNN要求序列等长，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。<br>
为此，__Encoder-Decoder结构先将输入数据编码成一个上下文向量c__：
![](https://pic3.zhimg.com/80/v2-03aaa7754bb9992858a05bb9668631a9_hd.jpg)

得到c有多种方式，最简单的方法就是把Encoder的最后一个隐状态赋值给c，还可以对最后的隐状态做一个变换得到c，也可以对所有的隐状态做变换。<br>
拿到c之后，__就用另一个RNN网络对其进行解码__，这部分RNN网络被称为Decoder。具体做法就是将c当做之前的初始状态h0输入到Decoder中：

![](https://pic3.zhimg.com/80/v2-77e8a977fc3d43bec8b05633dc52ff9f_hd.jpg)

还有一种做法是将c当做每一步的输入：
![](https://pic1.zhimg.com/80/v2-e0fbb46d897400a384873fc100c442db_hd.jpg)

由于这种Encoder-Decoder结构不限制输入和输出的序列长度，因此应用的范围非常广泛，比如：
* 机器翻译。Encoder-Decoder的最经典应用，事实上这一结构就是在机器翻译领域最先提出的。
* 文本摘要。输入是一段文本序列，输出是这段文本序列的摘要序列。
* 阅读理解。将输入的文章和问题分别编码，再对其进行解码得到问题的答案。
* 语音识别。输入是语音信号序列，输出是文字序列。

**6&nbsp;&nbsp;&nbsp;&nbsp;Attention机制**<br>
在Encoder-Decoder结构中，Encoder把所有的输入序列都编码成一个统一的语义特征c再解码，因此， c中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。如机器翻译问题，当要翻译的句子较长时，一个c可能存不下那么多信息，就会造成翻译精度的下降。<br>
Attention机制通过在每个时间输入不同的c来解决这个问题，下图是带有Attention机制的Decoder：
![](https://pic4.zhimg.com/80/v2-8da16d429d33b0f2705e47af98e66579_hd.jpg)

每一个c会自动去选取与当前所要输出的y最合适的上下文信息。具体来说，我们用 a_{ij} 衡量Encoder中第j阶段的hj和解码时第i阶段的相关性，最终Decoder中第i阶段的输入的上下文信息 c_i 就来自于所有 h_j 对 a_{ij} 的加权和。

以机器翻译为例（将中文翻译成英文）：

![](https://pic1.zhimg.com/80/v2-d266bf48a1d77e7e4db607978574c9fc_hd.jpg)

输入的序列是“我爱中国”，因此，Encoder中的h1、h2、h3、h4就可以分别看做是“我”、“爱”、“中”、“国”所代表的信息。在翻译成英语时，第一个上下文c1应该和“我”这个字最相关，因此对应的 a_{11} 就比较大，而相应的 a_{12} 、 a_{13} 、 a_{14} 就比较小。c2应该和“爱”最相关，因此对应的 a_{22} 就比较大。最后的c3和h3、h4最相关，因此 a_{33} 、 a_{34} 的值就比较大。

至此，关于Attention模型，我们就只剩最后一个问题了，那就是：__这些权重 a_{ij} 是怎么来的__？

事实上， a_{ij} 同样是从模型中学出的，它实际和Decoder的第i-1阶段的隐状态、Encoder第j个阶段的隐状态有关。

同样还是拿上面的机器翻译举例， a_{1j} 的计算（此时箭头就表示对h'和 h_j 同时做变换）：

![](https://pic1.zhimg.com/80/v2-5561fa61321f31113043fb9711ee3263_hd.jpg)

a_{2j} 的计算:

![](https://pic4.zhimg.com/80/v2-50473aa7b1c20d680abf8ca36d82c9e4_hd.jpg)

a_{3j} 的计算：

![](https://pic3.zhimg.com/80/v2-07f7411c77901a7bd913e55884057a63_hd.jpg)

以上就是带有Attention的Encoder-Decoder模型计算的全过程。

## TensorFlow中RNN实现的正确打开方式
**1&nbsp;&nbsp;&nbsp;&nbsp;学习单步的RNN：RNNCell**

如果要学习TensorFlow中的RNN，第一站应该就是去了解“[RNNCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/RNNCell)”，它是TensorFlow中实现RNN的基本单元，每个RNNCell都有一个call方法，使用方式是：(output, next_state) = call(input, state)。

在代码实现上，RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类BasicRNNCell和BasicLSTMCell。顾名思义，前者是RNN的基础类，后者是LSTM的基础类。这里推荐大家阅读其源码实现，一开始并不需要全部看一遍，只需要看下RNNCell、BasicRNNCell、BasicLSTMCell这三个类的注释部分，应该就可以理解它们的功能了。

除了call方法外，对于RNNCell，还有两个类属性比较重要：
* state_size
* output_size

前者是隐层的大小，后者是输出的大小。比如我们通常是将一个batch送入模型计算，设输入数据的形状为(batch_size, input_size)，那么计算时得到的隐层状态就是(batch_size, state_size)，输出就是(batch_size, output_size)。

可以用下面的代码验证一下（注意，以下代码都基于TensorFlow最新的1.2版本）：
```python
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = cell.call(inputs, h0) #调用call函数

print(h1.shape) # (32, 128)
```
