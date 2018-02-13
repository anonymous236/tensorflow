# 关于Kaggle
* Kaggle地址: https://www.kaggle.com/
* Kaggle平台是著名的数据分析竞赛在线平台。
<br>

# 关于Titanic之灾
* 问题链接: https://www.kaggle.com/c/titanic
* 问题描述:
  * 就是那个大家都熟悉的『Jack and Rose』的故事，豪华游艇倒了，大家都惊恐逃生，可是救生艇的数量有限，无法人人都有，
  副船长发话了『lady and kid first！』，所以是否获救其实并非随机，而是基于一些背景有rank先后的。
  * 训练和测试数据是一些乘客的个人信息以及存活状况，要尝试根据它生成合适的模型并预测其他人的存活状况。
  * 这是一个二分类问题，是logistic regression所能处理的范畴。最终要训练一个分类器，可以是SVM、神经网络、随机森林等模型。
* Titanic问题数据下载网址: https://www.kaggle.com/c/titanic/data
<br>

# 解决Titanic问题的思路：
* 首先采用正规化操作等手段对原始数据进行预处理
  <br>正规化(Normalization): 将主要特征字段转换为数值化的表达形式，这样才可作为分类器的输入。并且将数值都归一化到[0,1]的取值范围内，
  例如年龄字段值域为[0,100)，归一化可以将数值除以100。还要补齐缺失数据。
* 挑选特征向量的维度，以此训练一个分类器
  <br>维度过多、噪声过多会造成维度灾难(curse of dimensionality)
* 用训练好的分类器来预测数据结果
<br>

# 代码过程
```
代码过程可以分为: 数据读入及预处理 -> 构建计算图 -> 构建训练迭代过程 -> 执行训练 -> 存储模型 -> 预测测试数据结果
<br>完整代码版本: https://github.com/wangchen1ren/Titanic
```
