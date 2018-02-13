# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# 数据读入及预处理
data = pd.read_csv('./data-titanic/train.csv')

data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data = data.fillna(0)
dataset_X = data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_X = dataset_X.as_matrix()

data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased','Survived']]
dataset_Y = dataset_Y.as_matrix()

X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y, test_size = 0.2, random_state = 42)

print X_train.shape

# 构建计算图

#声明数据占位符
X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])

weights = tf.Variable(tf.random_normal([6, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')

# 构造前向传播计算图
y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

# 声明代价函数
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

# 声明优化算法
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# 声明计算准确率
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 构建训练迭代过程

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
    
