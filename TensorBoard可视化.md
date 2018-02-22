# TensorBoard可视化
TensorBoard的工作方式是启动一个Web服务，该服务进程从TensorBoard程序执行所得的事件日志文件(event files)中读取概要(summary)数据，然后将数据在网页中绘制成可视化图标。
## TensorBoard安装
```
pip install tensorboard
```
## 启动TensorBoard
* 问题描述：在terminal中输入tensorboard时提示找不到命令
* 解决方法：启动的时候需要换一种方式
  * 输入 *pip show tensorflow* 或者 *pip show tensorflow-gpu* 找到tensorflow的安装路径
  * 再输入  *python 安装路径/tensorflow/tensorboard/tensorboard.py --logdir=日志文件夹路径*
## 记录事件数据完整代码
```
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

X_train, X_val, y_train, y_val = train_test_split(dataset_X,
 dataset_Y, test_size = 0.2, random_state = 42)

# 构建计算图

# 声明数据占位符
# 用tf.name_scope限制命名空间，可在TensorBoard中展示节点名称
# 例如，在如下的with中，节点会命名为input/XXX
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 6])
    y = tf.placeholder(tf.float32, shape=[None, 2])

with tf.name_scope('classifier'):
    weights = tf.Variable(tf.random_normal([6, 2]), name='weights')
    bias = tf.Variable(tf.zeros([2]), name='bias')

    # 构造前向传播计算图
    y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

    # 添加直方图参数summary记录算子
    # tf.summary.histogram('weights', weights)
    #tf.scalar_summary('weights', weights)
    # tf.summary.histogram('bias', bias)
    #tf.scalar_summary('bias', bias)

# 声明代价函数
with tf.name_scope('cost'):
    cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
    cost = tf.reduce_mean(cross_entropy)

    # 添加 损失代价 标量summary
    # tf.summary.scalar('loss', cost)
    tf.scalar_summary('loss', cost)

# 声明优化算法
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# 声明计算准确率
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 添加 准确率 标量summary
    # tf.summary.scalar('accurancy', acc_op)
    tf.scalar_summary('accurancy', acc_op)

# 存储参数模型
saver = tf.train.Saver()

# 构建训练迭代过程

with tf.Session() as sess:
    # 创建summary写入操作
    # TensorBoard可通过命令 tensorboard --logdir=./logs 来启动
    # writer = tf.summary.FileWriter('./logs', sess.graph)
    writer = tf.train.SummaryWriter('./logs', sess.graph)
    # 合并所有summary算子
    # merged = tf.summary.merge_all()
    merged = tf.merge_all_summaries()

    tf.initialize_all_variables().run()
    for epoch in range(10):
        total_loss = 0
	for i in range(len(X_train)):
	    feed = {X: [X_train[i]], y: [y_train[i]]}
	    _, loss = sess.run([train_op, cost], feed_dict=feed)
	    total_loss += loss
	print "Epoch: %d, total loss=%.9f" % (epoch+1, total_loss)

        summary, accurancy = sess.run([merged, acc_op], feed_dict={X: X_val,y:y_val})
        writer.add_summary(summary, epoch)
    print "Training completed!"

    save_path = saver.save(sess, "./model.ckpt")
    
with tf.Session() as sess2:
    saver.restore(sess2, "./model.ckpt")
    # 训练测试集验证
    pred = sess2.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred,1), np.argmax(y_val,1))
    np_accurancy = np.mean(correct.astype(np.float32))
    print "Numpy accurancy on validation set: %.9f" % np_accurancy

    accurancy = sess2.run(acc_op, feed_dict={y_pred:pred, y:y_val})
    print "Accurancy on validation set: %.9f" % accurancy

    # 进行预测
    testdata = pd.read_csv('./data-titanic/test.csv')
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    testdata = testdata.fillna(0)
    testdataset_X = testdata[['Sex','Age','Pclass','SibSp','Parch','Fare']]
    
    predictions = np.argmax(sess2.run(y_pred, feed_dict={X: testdataset_X}),1)
    data_predictions = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": predictions
    })
    # data_predictions.to_csv("./titanic-predictions.csv", index=False)
```
生成的事件日志文件会是一个以"events."开头的文件，里面的内容都是使用protocol buffer序列化之后的二进制数据，只能通过TensorBoard打开。
