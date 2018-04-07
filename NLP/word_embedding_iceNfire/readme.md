## py说明
 * [word_embedding_iceNfire.py](https://github.com/anonymous236/tensorflow/blob/master/NLP/word_embedding_iceNfire/word_embedding_iceNfire.py)
 * names_separate.txt 文本的姓名停用词
 
## jieba分词、使用word2vec训练模型-skip-gram模型
 * Word2Vec是从大量文本语料中以无监督的方式学习语义知识的一种模型<br>通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。Embedding其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在空间嵌入到一个新的空间中去。
 * Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。而CBOW是给定上下文，来预测input word
![skip-gram&cbow](https://github.com/anonymous236/figure/blob/master/skip-gram%26cbow.jpg)

## 代码过程
```python
from tqdm import tqdm
import jieba
# 调用 jieba分词module后，添加单词本（人名等）:
jieba.load_userdict("data/names_separate.txt")
import sys
```
### 数据处理和准备阶段
```python
# 输入原始文档(《冰与火之歌》)
filename = 'data/ice_and_fire_utf8.txt'
text_lines = []

with open(filename, 'r') as f:
    for line in tqdm(f):
        text_lines.append(line)
print('总共读入%d行文字'% (len(text_lines)))

# 调用jieba分词工具将句子切分成词语的序列
data_words, data_lines = [], []
# data_words: 训练我们的cbow和skip-gram模型
# data_lines: 调用gensim.word2vec训练word2vec模型

## 分词:
for line in tqdm(text_lines):
    one_line = [' '.join(jieba.cut(line, cut_all=False))][0].split(' ')
    data_words.extend(one_line)
    data_lines.append(one_line)
    
# 去掉标点和数字
import re
# 标点符号 (punctuation)
punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．＊：；Ｏ？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…０１２３４５６７８９''')
isNumber = re.compile(r'\d+.*')

filter_words = [w for w in data_words if (w not in punct)
                and (not isNumber.search(w.lower()))]

# 建立词典
from collections import Counter

vocabulary_size = 30000
def build_vocab(words):
    """对文字数据中最常见的单词建立词典
    
    Arguments:
        words: 一个list的单词,来自分词处理过的文本数据库.
    
    Returns:
        data: 输入words数字编码后的版本
        count: dict, 单词 --> words中出现的次数
        dictionary: dict, 单词 --> 数字ID的字典
        reverse_dictionary: dict, 数字ID-->单词的字典
    """
    # 1. 统计每个单词的出现次数
    words_counter = Counter(filter_words)
    # 2. 选取常用词
    count = [['UNK', -1]]
    count.extend(words_counter.most_common(vocabulary_size - 1))
    # 例如，count执行完后为：[['UNK', -1], ('a', 2), ('b', 1), ('c', 1)]
    
    # 3. 词语编号
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 例如，dictionary执行完后为：{'UNK': 0, 'a': 1, 'b': 2, 'c': 3}
    
    data = list()
    # 4. 引入特殊词语UNK
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    # 将dictionary中存在的词的编号写入data中，不存在写0
    
    print(unk_count)
    count[0][1] = unk_count
    # 例如，count执行完后为：[['UNK', 41973], ('a', 2), ('b', 1), ('c', 1)]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # 将key和value互换，例如变为：{0: 'UNK', 1: 'a', 2: 'b', 3: 'c'}
    return data, count, dictionary, reverse_dictionary
# 生成词典
data, count, dictionary, reverse_dictionary = build_vocab(filter_words)

# 验证单词到数字的编码是正确的

demo_num = data[1000:1100]
print(demo_num)
demo_str = ''
for i in range(1000, 1100):
    demo_str = demo_str+(reverse_dictionary[data[i]])+' '
print(demo_str)
```
```shell
ubuntu:~$ [2691, 195, 655, 778, 11, 1, 715, 6580, 19, 88, 11, 65, 3925, 3318, 510, 51, 137, 1, 3270, 232, 0, 847, 2691, 5, 533, 7008, 1019, 385, 51, 1, 2301, 7, 7240, 1, 449, 51, 86, 32, 11374, 11375, 362, 5857, 1, 2926, 2822, 711, 859, 11, 1457, 10252, 573, 106, 622, 1653, 5, 1617, 1, 2118, 13644, 2118, 1523, 778, 11, 119, 9374, 2118, 4724, 706, 1, 4224, 8976, 11376, 1160, 1160, 131, 5858, 517, 1, 15870, 7, 1554, 1, 29358, 51, 573, 106, 14671, 5, 231, 2220, 11, 860, 1, 3050, 232, 7, 711, 2691, 113, 1]
          征服 之 路 尽管 他们 的 军队 数量 不 多 他们 却 有着 西方 世界 中 最后 的 三条 龙 UNK 此 征服 了 整个 大陆 七大 王国 中 的 六个 在 最初 的 战争 中 便 被 降服 唯独 多恩 激烈 的 反抗 以至于 伊耿 同意 他们 保持 独立 坦格利安 家族 同样 放弃 了 原来 的 信仰 改为 信仰 七神 尽管 他们 还是 违背 信仰 按照 瓦雷利亚 的 传统 兄妹 通婚   并 遵守 维斯特洛 的 风俗 在 接下来 的 数十年 中 坦格利安 家族 扑灭 了 所有 反对 他们 统治 的 叛乱 龙 在 伊耿 征服 后 的
```
### 使用word2vec训练模型
```python
import gensim
model = gensim.models.Word2Vec(iter=1, min_count=0)
# 没有min_count字段会报“RuntimeError: you must first build vocabulary before training the model”的错

model.build_vocab(data_lines)
model.train(data_lines, total_examples = len(data_lines), epochs = 10)
test_words = ['史塔克', '提利昂', '琼恩', '长城', '衣物', '力量', '没关系']
neighbors = []
for test_word in test_words:
    neighbors.append(model.most_similar(test_word))
    for i in range(len(neighbors)):
    str = ' '.join([x[0] for x in neighbors[i]])
    print('%s:' % test_words[i])
    print('\t%s\n' % (str))
```
```shell
ubuntu:~$ 史塔克:
              	徒利 波顿 艾林 葛雷乔伊 提利尔 兰尼斯特 卢斯 拉萨 戴林恩 佛雷
          ...
          没关系:
              	受不了 一辈子 不行 没用 无所谓 没差 不成问题 不怕 死活 用不着
```
### skip-gram模型
```python
import numpy as np
import tensorflow as tf
import collections

data_index = 0
# 设置相同的seed，每次生成的随机数相同
np.random.seed(0)

# 只是简答的数组记忆功能。例如[1,2,3,4,5]会记忆“3”左右长度为2的数字为：“1、2、4、5”
# batch_size为批大小，程序每次训练的数量；为了凑齐batch_size个组合，我们需要batch_size//num_skips个滑动窗，见以下for循环
# slide_window为记忆的左右宽度
# num_skips为“目标单词”要被记住的数量；num_skips <= 2 * slide_window
''' 
num_skips = 2, slide_window = 1时，程序的最后输出为：
7 --> 9
7 --> 8  # 7记住了2次，因为num_skips = 2
3 --> 5
3 --> 4
1 --> 2
1 --> 9
9 --> 8
9 --> 2
'''
def generate_batch_sg(data, batch_size, num_skips, slide_window):
    
    global data_index # 使用全局变量
    # assert断言，用来测试表示式，其返回值为假，就会触发异常
    assert batch_size % num_skips == 0  # 为了凑齐batch_size个组合，需要batch_size//num_skips个滑动窗，必须为整数
    assert num_skips <= 2 * slide_window  # 为“目标单词”要被记住的数量
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 用来存放“目标单词”
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # 用来存放 “目标单词” 的左右邻居单词
    
    # 滑动窗：[ slide_window target slide_window ]，宽度为 span
    # 每个单词的左slide_window和右slide_window被当作它的语境
    span = 2 * slide_window + 1  # 一个单词和它的语境的总长度
    # deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈；总长度 maxlen=span
    buffer = collections.deque(maxlen=span)
    
    # 扫过文本，将一个长度为 2*slide_window+1 的滑动窗内的词语放入buffer
    # buffer里面，居中的是target，“当前单词”。（将滑动窗里的单词记录就是它的邻居单词，见下边的代码）
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    
    # # 下面的 for 循环：
    # 在产生一个有batch_size个样本的minibatch的时候，
    # 我们选择 batch_size//num_skips 个 “当前单词” （或者叫“目标单词”）
    # 并且从“当前单词”左右的2*slide_window 个词语组成的context里面选择num_skips个单词
    # “当前单词”（我们叫做x）和num_skips个单词中的每一个（我们叫做y_i）组成一个监督学习样本：
    # 给定单词x, 在它的context里面应该大概率地出现单词y_i
    for i in range(batch_size // num_skips):
        # 在每个长度为2*slide_window + 1 的滑动窗里面，
        # 我们选择num_skips个（“当前单词”，“语境单词”）的组合
        # 为了凑齐batch_size个组合，我们需要batch_size//num_skips个滑动窗
        
        # 这里的rand_x是一个长度为span的随机整数列表，表示滑动窗内的span个单词的下标
        rand_x = np.random.permutation(span)
        j, k = 0, 0
        # j从0开始，表示从头开始遍历滑动窗口
        for j in range(num_skips):
            while rand_x[k]==slide_window:
                # 这里的意思是，我们想要找到“目标单词”的左右邻居单词，那么它本身就可以省略了
                k += 1
            batch[i * num_skips + j] = buffer[slide_window]  # buffer[slide_window]表示“目标单词”就是滑动窗口中间的那个单词
            labels[i * num_skips + j, 0] = buffer[rand_x[k]]  # 挨个记录滑动窗口内的邻居单词
            k += 1
        
        # 将滑动窗向右滑动随机步
        rand_step = np.random.randint(1,5)
        for _ in range(rand_step):
        	# buffer是一个列表，只能存储最新的总长度为 maxlen=span的数据
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        
    return batch, labels

'''
data=[9,8,7,6,5,4,3,2,1]
# 测试代码：
batch_size = 8
for num_skips, slide_window in [(2, 1), (4,2)]:
    batch, labels = generate_batch_sg(data = data,
                                      batch_size=batch_size,
                                      num_skips=num_skips,
                                      slide_window=slide_window)
    for i in range(batch_size):
        print('%s --> %s' % (batch[i],labels[i][0]))
'''
```
### 构建模型、进行训练
```python
import random
import math

batch_size = 128
num_sampled = 64 # 近似计算cross entropy loss的时候的negative examples参数
embedding_size = 128 # Dimension of the embedding vector

# skip-gram的两个重要的hyperparameters:
# 1. 语境范围：考虑和周围多少个词语的共存关系
slide_window = 1
# 2.　”使用频率“：（当前单词，临近单词）的组合使用多少次
num_skips = 1 # How many times to reuse an input to generate a label

# 产生测试数据
valid_size = 8
valid_examples = list(np.random.permutation(1000)[:valid_size])
names = ['史塔克', '提利昂', '琼恩', '长城', '南方', '死亡', '家族', '支持', '愤怒']
for name in names:
    valid_examples.append(dictionary[name])
    valid_size += 1
    

graph_sg = tf.Graph()

with graph_sg.as_default():#, tf.device('/cpu:0'):

    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])  # 对应着模型的输入
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])  # 每个单词周围的context，对应着模型的输出
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    # 要学习的词向量参数
    embeddings = tf.Variable(
        # 均匀分布随机数，范围为[minval,maxval]
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # softmax分类器
    softmax_weights = tf.Variable(
        # 截断正态分布随机数，均值mean,标准差stddev
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    # 选取一个tensor里面索引对应的元素
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities 
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate = 3.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # 对词向量进行归一化；除以其L2范数后得到标准化后的normalized_embeddings
    normalized_embeddings = embeddings / norm
    # 根据校验集合，查找出相应的词向量
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    # 计算cosine相似度
    # tf.matmul是矩阵乘法；tf.transpose对矩阵进行转换操作
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    
    
num_steps = 100001

with tf.Session(graph=graph_sg) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch_sg(
            data, batch_size, num_skips, slide_window)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            # eval():将字符串str当成有效的表达式来求值并返回计算结果
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                # （-x）.argsort():将x中的元素从大到小排列，提取其对应的index(索引)
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings_sg = normalized_embeddings.eval()
```
```shell
ubuntu:~$ Average loss at step 30000: 3.418303
          Nearest to 玖健: 杰斯, 梅拉, 睡过去, 朔风, 葛兰, 何相干, 抚养, 稳稳地,
          Nearest to 而言: 雪中送炭, 惨死, 地举, 推着, 怕, 矫健, 乐, 坐回,
          Nearest to 打: 飘, 哭, 誓, 讨回, 之口, 之红剑, 从, 幼弟,
          Nearest to 一半: 倒不如, 飞石, 三匹, 石像, 大不相同, 未婚夫, 凯岩, 一部分,
          Nearest to 随后: 是否是, 坐镇, 门卫, 克利夫, 蹲下来, 立刻, 圈, 不断,
          Nearest to 结婚: 届时, 莱摩儿, 去世, 挨饿, 回来, 竞赛, 到来, 再试一次,
          Nearest to 会: 能, 不会, 可能, 告, 足以, 该, 它们, 奸,
          Nearest to 其他: 别的, 不少, 们, 两个, 很多, 墨, 任何, 复国,
          Nearest to 史塔克: 波顿, 兰尼斯特, 佛罗伦, 热汤, 坦格利安, 毕斯柏里, 横飞, 徒利,
          Nearest to 提利昂: 瑟曦, 詹姆, 布蕾妮, 丹妮, 昆廷, 小指头, 奈德, 太后,
          Nearest to 琼恩: 雪诺, 席恩, 热气腾腾, 霍斯特, 乔斯, 山姆, 丹妮, 弗塔,
          Nearest to 长城: 黄昏, 奔流, 谛听, 接收, 哪里, 老派, 君临, 跳舞,
          Nearest to 南方: 北方, 维斯, 贝勒, 君临, 哪里, 奔流, 原地, 赫伦堡,
          Nearest to 死亡: 想方设法, 恶名昭彰, 名望, 美蕊, 指挥权, 这么回事, 梭尔, 士,
          Nearest to 家族: 家, 军, 爵士, 伯爵, 台前, 夫人, 全数, 半小时,
          Nearest to 支持: 答应, 聚会, 心树前, 好家伙, 清白, 兑水, 史塔克, 科布,
          Nearest to 愤怒: 减轻, 苦涩, 流脓, 歪歪扭扭, 我准, 明智, 最高, 不安,
          ...
```
