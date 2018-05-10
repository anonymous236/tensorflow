# -*- coding:utf-8 -*-
import random
import numpy as np

def build_data(n):
    xs = []
    ys = []
    for i in range(2000):
	k = random.uniform(1,50)
	x = [[np.sin(k + j)] for j in range(0,n)]
	y = [np.sin(k + n)]
	xs.append(x)
	ys.append(y)
    train_x = np.array(xs[:1500])
    train_y = np.array(ys[:1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])
    return (train_x, train_y, test_x, test_y)

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

length = 10
time_step_size = length
vector_size = 1
batch_size = 10
test_size = 10

# build data
(train_x, train_y, test_x, test_y) = build_data(length)
print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# 使用placeholder声明输入占位符，第一个维度是batch_size，None表示任意，从第二个维度开始才是特征维度
# 输入X的形状:[batch_size,time_step_size,vector_size]
X = tf.placeholder("float", [None, time_step_size, vector_size])
Y = tf.placeholder("float", [None, 1])

W = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
B = tf.Variable(tf.random_normal([1], stddev=0.01))

def seq_predict_model(X, w, b, time_step_size, vector_size):
    # 数组转置函数
    # X转为:[time_step_size,batch_size,vector_size]
    X = tf.transpose(X, [1,0,2])
    # 调整tensor X的维度  -1表示不指定维度
    # X最终的shape为:[time_step_size*batch_size, vector_size]
    X = tf.reshape(X, [-1, vector_size])
    # 以第0维度，把X分为time_step_size份，切分后的shape为[batch_size, vector_size]
    X=tf.split(X,time_step_size,0)

    cell = core_rnn_cell.BasicRNNCell(num_units = 10)
    # state_size为隐层的大小，即为10
    initial_state=tf.zeros([batch_size,cell.state_size])
    outputs,_states=core_rnn.static_rnn(cell,X,initial_state=initial_state)

    return tf.matmul(outputs[-1],w)+b,cell.state_size

pred_y, _ = seq_predict_model(X,W,B,time_step_size,vector_size)


#声明代价函数
loss=tf.square(tf.subtract(Y,pred_y))
#加入优化算法
train_op=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#构造训练迭代过程并预测测试数据结果
with tf.Session() as sess:
    #初始化所有变量，必须最先执行
    tf.initialize_all_variables().run()
    #以下为训练迭代，迭代50轮
    for i in range(50):
        for end in range(batch_size,len(train_x),batch_size):
            begin=end-batch_size
            x_value=train_x[begin:end]
            y_value=train_y[begin:end]
            sess.run(train_op,feed_dict={X:x_value,Y:y_value})

        # 在训练的过程中开始测试
        test_indices=np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices=test_indices[0:test_size]
        x_value=test_x[test_indices]
        y_value=test_y[test_indices]
        # 使用均方差作为代价函数
        val_loss=np.mean(sess.run(loss,feed_dict={X:x_value,Y:y_value}))
        print 'Run %s: '%i,val_loss
