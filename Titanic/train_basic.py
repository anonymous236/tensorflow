# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# 数据读入及预处理
data = pd.read_csv('./data-titanic/train.csv')

data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
# 填充缺失字段
data = data.fillna(0)
dataset_X = data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_X = dataset_X.as_matrix()

data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased','Survived']]
dataset_Y = dataset_Y.as_matrix()

#sklearn库提供用于切分数据集的工具函数 train_test_split
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y, test_size = 0.2, random_state = 42)


# 构建计算图

#声明数据占位符
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])

# 变量定义，构造全零tensor: zeros()、 随机正态分布tensor: random_normal()
weights = tf.Variable(tf.random_normal([6, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')

# 构造前向传播计算图
y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

# 声明代价函数
'''
解决计算交叉熵中存在的log(0)问题:
* 在计算log()时，直接加入一个极小的误差值，使计算合法。这样可以避免计算log(0)，但存在的问题是加入误差后的计算值会突破1
* 使用clip()函数，当y_pred接近0时，将其赋值成为极小误差值，如范围为[1e-10,1]
* 当计算交叉熵出现NAN时，显式地将cost设置为0。
'''
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

# 声明优化算法
# 优化器内部会自动构建梯度计算和反向传播部分的计算图。使用随机梯度下降算法：
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# 声明计算准确率
# tf.argmax()返回最大值所在的坐标，参数 axis=1 表示以“行”为标准
# tf.cast()为类型转换函数
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 构建训练迭代过程
'''
Session.run()有两个关键的参数：fetches 和 feed_dict
* fetches指定需要被计算的节点，节点可以是算子op，也可是tensor
* feed_dict指定计算所需要的输入数据，传入一个字典: {key:value}，其中key为输入占位符placeholder，value为真实的输入数据
'''

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for epoch in range(10):
        total_loss = 0
	for i in range(len(X_train)):
	    feed = {X: [X_train[i]], y: [y_train[i]]}
	    _, loss = sess.run([train_op, cost], feed_dict=feed)
	    total_loss += loss
	print "Epoch: %d, total loss=%.9f" % (epoch+1, total_loss)
    print "Training completed!"
    
    # 训练测试集验证
    pred = sess.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred,1), np.argmax(y_val,1))
    np_accurancy = np.mean(correct.astype(np.float32))
    print "Numpy accurancy on validation set: %.9f" % np_accurancy

    accurancy = sess.run(acc_op, feed_dict={y_pred:pred, y:y_val})
    print "Accurancy on validation set: %.9f" % accurancy

    # 进行预测
    testdata = pd.read_csv('./data-titanic/test.csv')
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    testdata = testdata.fillna(0)
    testdataset_X = testdata[['Sex','Age','Pclass','SibSp','Parch','Fare']]
    
    predictions = np.argmax(sess.run(y_pred, feed_dict={X: testdataset_X}),1)
    data_predictions = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": predictions
    })
    data_predictions.to_csv("./titanic-predictions.csv", index=False)
    
