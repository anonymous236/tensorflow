## 以TextCNN为例学习CNN
* TextCNN 是利用卷积神经网络对文本进行分类的算法，由 Yoon Kim 在 [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) 一文
中提出. 是2014年的算法.<br>
* CNN 的主要过程如下:
```
![](https://github.com/anonymous236/tensorflow/blob/master/CNN/cnn1.png)
![](https://github.com/anonymous236/tensorflow/blob/master/CNN/cnn2.png)
```
## 解读TextCNN
* TextCNN源码 [https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
* 以下转载至 [http://www.dataguru.cn/forum.php?mod=viewthread&tid=637971&extra=page=1&page=1](http://www.dataguru.cn/forum.php?mod=viewthread&tid=637971&extra=page=1&page=1)

**1&nbsp;&nbsp;&nbsp;这个class的主要作用是什么?**<br>
TextCNN类搭建了一个最basic的CNN模型，有input layer，convolutional layer，max-pooling layer和最后输出的softmax layer.<br>
但是又因为整个模型是用于文本的(而非CNN的传统处理对象：图像)，因此在cnn的操作上相对应地做了一些小调整：
* 对于文本任务，输入层自然使用了word embedding来做input data representation
* 接下来是卷积层，大家在图像处理中经常看到的卷积核都是正方形的，比如4\*4，然后在整张image上沿宽和高逐步移动进行卷积操作。但是nlp中输入的"image"是一个词矩阵，比如n个words，每个word用200维的vector表示的话，这个"image"就是n\*200的矩阵，卷积核只在高度上已经滑动，在宽度上和word vector的维度一致（=200），也就是说每次窗口滑动过的位置都是完整的单词，不会将几个单词的一部分"vector"进行卷积，这也保证了word作为语言中最小粒度的合理性。（当然，如果研究的粒度是character-level而不是word-level，需要另外的方式处理）
* 由于卷积核和word embedding的宽度一致，一个卷积核对于一个sentence，卷积后得到的结果是一个vector， shape=（sentence_len - filter_window + 1, 1），那么，在max-pooling后得到的就是一个Scalar.所以，这点也是和图像卷积的不同之处，需要注意一下
* 正是由于max-pooling后只是得到一个scalar，在nlp中，会实施多个filter_window_size（比如3,4,5个words的宽度分别作为卷积的窗口大小），每个window_size又有num_filters个（比如64个）卷积核。一个卷积核得到的只是一个scalar太孤单了，智慧的人们就将相同window_size卷积出来的num_filter个scalar组合在一起，组成这个window_size下的feature_vector
* 最后再将所有window_size下的feature_vector也组合成一个single vector，作为最后一层softmax的输入<br>
  **一个卷积核对于一个句子，convolution后得到的是一个vector；max-pooling后，得到的是一个scalar**
  
总结一下这个类的作用就是：搭建一个用于文本数据的CNN模型！

**2&nbsp;&nbsp;&nbsp;模型参数**
* 关于model
  * filter_sizes: 3,4,5, Comma-separated filter sizes (default: '3,4,5')
  * num_filters: 128, Number of filters per filter size (default: 128)
  * dropout_keep_prob: 0.5, Dropout keep probability (default: 0.5)
  * l2_reg_lambda: 0.0, L2 regularization lambda (default: 0.0)
* 关于training
  * batch_size: 64, Batch Size (default: 64)
  * num_epochs: 200, Number of training epochs (default: 200)
  * evaluate_every: 100, Evaluate model on dev set after this many steps (default: 100)
  * checkpoint_every: 100, Save model after this many steps (default: 100)
  * num_checkpoints: 5, Number of checkpoints to store (default: 5)
  
**3&nbsp;&nbsp;&nbsp;Dropout**
* 正则是解决过拟合的问题，在最后一层softmax的时候是full-connected layer，因此容易产生过拟合.
* 策略就是在: 
  * 在**训练**阶段，对max-pooling layer的输出实行一些dropout，以概率p激活，激活的部分传递给softmax层.<br>
  * 在**测试**阶段，w已经学好了，但是不能直接用于unseen sentences，要乘以p之后再用，这个阶段没有dropout了全部输出给softmax层.
  
**4&nbsp;&nbsp;&nbsp;Embedding Layer**
```python
# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        name="W")
    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
```
存储全部word vector的矩阵W初始化时是随机random出来的，也就是paper中的第一种模型CNN-rand.<br>
训练过程中并不是每次都会使用全部的vocabulary，而只是产生一个batch（batch中都是sentence，每个sentence标记了出现哪些word(较大长度为sequence_length)，因此batch相当于一个二维列表），这个batch就是input_x.
```python
self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
```
tf.nn.embedding_lookup:查找input_x中所有的ids，获取它们的word vector。batch中的每个sentence的每个word都要查找。所以得到的embedded_chars的shape应该是[None, sequence_length, embedding_size]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1)<br>
输入的word vectors得到之后，下一步就是输入到卷积层，用到 [tf.nn.conv2d](https://blog.csdn.net/mao_xiao_feng/article/details/78004522) 函数
  ```python
  tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
  ```
> 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
>> input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一<br><br>
>> filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维<br><br>
>> strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4<br><br>
>> padding： string类型的量，只能是"SAME", "VALID"其中之一，这个值决定了不同的卷积方式<br><br>
>> use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true<br><br>
结果返回一个Tensor，这个输出，就是我们常说的feature map
