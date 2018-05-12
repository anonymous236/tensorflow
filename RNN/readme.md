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

可以用下面的代码验证一下（注意，以下代码都基于TensorFlow的1.2版本）：
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

对于BasicLSTMCell，情况有些许不同，因为LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是(batch_size, state_size)的形状：
```python
import tensorflow as tf
import numpy as np
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.call(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
```

**2&nbsp;&nbsp;&nbsp;&nbsp;学习如何一次执行多步：tf.nn.dynamic_rnn**

基础的RNNCell有一个很明显的问题：对于单个的RNNCell，我们使用它的call函数进行运算时，只是在序列时间上前进了一步。比如使用x1、h0得到h1，通过x2、h1得到h2等。这样的h话，如果我们的序列长度为10，就要调用10次call函数，比较麻烦。对此，__TensorFlow提供了一个tf.nn.dynamic_rnn函数，使用该函数就相当于调用了n次call函数__。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。

具体来说，设我们输入数据的格式为(batch_size, time_steps, input_size)，其中time_steps表示序列本身的长度，如在Char RNN中，长度为10的句子对应的time_steps就等于10。最后的input_size就表示输入数据单个序列单个时间维度上固有的长度。另外我们已经定义好了一个RNNCell，调用该RNNCell的call函数time_steps次，对应的代码就是：
```python
# inputs: shape = (batch_size, time_steps, input_size) 
# cell: RNNCell
# initial_state: shape = (batch_size, cell.state_size)。初始状态。一般可以取零矩阵
outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
```
此时，得到的outputs就是time_steps步里所有的输出。它的形状为(batch_size, time_steps, cell.output_size)。state是最后一步的隐状态，它的形状为(batch_size, cell.state_size)。此处建议大家阅读 [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) 的文档做进一步了解。

**3&nbsp;&nbsp;&nbsp;&nbsp;学习如何堆叠RNNCell：MultiRNNCell**

很多时候，单层RNN的能力有限，我们需要多层的RNN。将x输入第一层RNN的后得到隐层状态h，这个隐层状态就相当于第二层RNN的输入，第二层RNN的隐层状态又相当于第三层RNN的输入，以此类推。在TensorFlow中，可以使用 [tf.nn.rnn_cell.MultiRNNCell](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py) 函数对RNNCell进行堆叠，相应的示例程序如下：
```python
import tensorflow as tf
import numpy as np

# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN
# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128
print(cell.state_size) # (128, 128, 128)
# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = cell.call(inputs, h0)
print(h1) # tuple中含有3个32x128的向量
```

**4&nbsp;&nbsp;&nbsp;&nbsp;可能遇到的坑1：Output说明**

在经典RNN结构中有这样的图：

![](https://pic3.zhimg.com/80/v2-71bec838a7a8410e477a186bf7cb2299_hd.jpg)

在上面的代码中，我们好像有意忽略了调用call或dynamic_rnn函数后得到的output的介绍。将上图与TensorFlow的BasicRNNCell对照来看。h就对应了BasicRNNCell的state_size。那么， __y是不是就对应了BasicRNNCell的output_size呢？答案是否定的__ 。

找到源码中BasicRNNCell的call函数实现：
```python
def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    output = self._activation(_linear([inputs, state], self._num_units, True))
    return output, output
```
这句“return output, output”说明 __在BasicRNNCell中，output其实和隐状态的值是一样的__ 。 因此， **`我们还需要额外对输出定义新的变换，才能得到图中真正的输出y`** 。由于output和隐状态是一回事，所以在BasicRNNCell中，state_size永远等于output_size。TensorFlow是出于尽量精简的目的来定义BasicRNNCell的，所以省略了输出参数，我们这里一定要弄清楚它和图中原始RNN定义的联系与区别。

再来看一下BasicLSTMCell的call函数定义（函数的最后几行）：
```python
new_c = (
    c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
new_h = self._activation(new_c) * sigmoid(o)

if self._state_is_tuple:
  new_state = LSTMStateTuple(new_c, new_h)
else:
  new_state = array_ops.concat([new_c, new_h], 1)
return new_h, new_state
```
我们只需要关注self.\_state\_is\_tuple == True的情况，因为self.\_state\_is\_tuple == False的情况将在未来被弃用。返回的隐状态是new\_c和new\_h的组合，而output就是单独的new\_h。如果我们处理的是分类问题，那么我们还需要对new\_h添加单独的Softmax层才能得到最后的分类概率输出。

**5&nbsp;&nbsp;&nbsp;&nbsp;可能遇到的坑2：因版本原因引起的错误**

在前面我们讲到堆叠RNN时，使用的代码是：
```python
# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN
```
这个代码在TensorFlow 1.2中是可以正确使用的。但在之前的版本中（以及网上很多相关教程），实现方式是这样的：
```python
one_cell =  tf.nn.rnn_cell.BasicRNNCell(num_units=128)
cell = tf.nn.rnn_cell.MultiRNNCell([one_cell] * 3) # 3层RNN
```
如果在TensorFlow 1.2中还按照原来的方式定义，就会引起错误！
