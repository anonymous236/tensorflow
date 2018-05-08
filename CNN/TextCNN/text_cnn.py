#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class TextCNN(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                # 所有词汇，每个词对应一个embedding_size的向量
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # 将input_x中的每句话的每一个词都用embedding_size维的向量来表示
            # 表示后的向量维度是：[input_x.shape[0], sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 因为卷积操作conv2d()需要输入的是四维数据，分别代表着批处理大小、宽度、高度、通道数。
            # 而embedded_chars只有前三维，所以需要添加一维，设为1。变为：[input_x.shape[0], sequence_length, embedding_size, 1]
            # [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 卷积层、池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                # 构建卷积核尺寸，输入和输出channel分别为1和num_filters
                # 相当于CNN中的卷积核，它要求是一个Tensor，
                # 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
                # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
                # 要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 矩阵内积 + 偏置 : W * X + b
                # W 就是卷积核
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    # [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
                    self.embedded_chars_expanded,
                    W,
                    # 卷积时在图像每一维的步长，这是一个一维的向量，长度4
                    strides=[1, 1, 1, 1],
                    # string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同的卷积方式
                    padding="VALID",
                    name="conv")
                # 做完卷积之后，矩阵大小为 [None, sequence_length - filter_size + 1, 1, num_filters]

                # 非线性操作，激活函数：relu(W*x + b)
                # h 是对卷积结果进行非线性转换之后的结果
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 最大池化, 选取卷积结果的最大值pooled的尺寸为[None, 1, 1, 128](卷积核个数)
                # 本质上是一个特征向量，最后一个维度是特征代表数量
                pooled = tf.nn.max_pool(
                    h, # 待池化的四维张量，维度是[batch, height, width, channels]
                    # 池化窗口大小，长度（大于）等于4的数组，与value的维度对应，
                    # 一般为[1,height,width,1]，batch和channels上不池化
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                # pooled_outputs最终为一个长度为3的列表。每一个元素都是[None,1,1,128]的Tensor张量
                # 对每个卷积核重复上述操作，故pooled_outputs的数组长度应该为len(filter_sizes)

        # Combine all the pooled features
        # 将所有window_size下的feature_vector也组合成一个single vector，作为最后一层softmax的输入
        # 因为3种filter卷积池化之后是一个scalar, 共有
        num_filters_total = num_filters * len(filter_sizes)
        # 对pooled_outputs在第四个维度上进行合并，变成一个[None,1,1,384]Tensor张量
        # 将不同核产生的计算结果（features）拼接起来
        # tf.concat(values, concat_dim)连接values中的矩阵，concat_dim指定在哪一维（从0计数）连接
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 把每一个max-pooling之后的张量合并起来之后得到一个长向量 [batch_size, num_filters_total]
        # 展开成两维Tensor[None,384]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 是cnn中最流行的正则化方法
        # dropout layer随机地选择一些神经元，使其失活。
        # 这样可以阻止co-adapting,迫使它们每一个都学习到有用的特征。
        # 失活的神经单元个数由dropout_keep_prob 决定。在训练的时候设为 0.5 ,测试的时候设为 1 (disable dropout)
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 全连接层计算输出向量(w*h+b)和预测(scores向量中的最大值即为预测结果)；其实是个softmax分类器
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 损失函数
        # Calculate Mean cross-entropy loss     计算scores和input_y的交叉熵损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        # Accuracy计算准确度，预测和真实标签相同即为正确
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
