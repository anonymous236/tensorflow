## FastText文章中使用的搜狗新闻数据集
* Categories
  + sports
  + finance
  + entertainment 
  + automobile
  + technology”

## 下载数据 ，预先处理
1. gbk --> utf-8
2. 从<url>提取类别标签
3. 从<content>提取文本内容

## 参考资料
* [word2vec 构建中文词向量](http://www.cnblogs.com/Newsteinwell/p/6034747.html)
* [Automatic Online News Issue Construction in Web Environment WWW2008](http://wwwconference.org/www2008/papers/pdf/p457-wang.pdf)

## 代码过程
* 这个实验采用搜狗实验室的搜狗新闻语料库，数据链接 http://www.sogou.com/labs/resource/cs.php
  + 1-1. 下载下来的文件名为： news_sohusite_xml.full.zip
  + 1-2. 解压 --> xvf news_sohusite_xml.dat
  + 1-3.A 字符编码转换  cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>\|<url>" > corpus_labeled.txt
  + 1-3.B 使用codec module读入gbk编码的原始文档，使用unicode或者编码为utf-8
  
```shell
corpus_labeled.txt 语料库格式如下：
<url>http://gongyi.sohu.com/20120706/n347457739.shtml</url>
<content>南都讯　记者刘凡　周昌和　任笑一　继推出日票后，深圳今后将设地铁ＶＩＰ 头等车厢，设坐票制。昨日，《南都ＭＥＴＲＯ》创刊仪式暨２０１２年深港地铁圈高峰论坛上透露，在未来的１１号线上将增加特色服务，满足不同消费层次的乘客的不同需求，如特设行李架的车厢和买双倍票可有座位坐的ＶＩＰ车厢等。论坛上，深圳市政府副秘书长、轨道交通建设办公室主任赵鹏林透露，地铁未来的方向将分等级，满足不同层次的人的需求，提供不同层次的有针对的服务。其中包括一些档次稍微高一些的服务。“我们要让公共交通也能满足档次稍高一些的服务”。比如，尝试有座位的地铁票服务。尤其是一些远道而来的乘客，通过提供坐票服务，让乘坐地铁也能享受到非常舒适的体验。他说，这种坐票的服务有望在地铁３期上实行，将加挂２节车厢以实施花钱可买座位的服务。“我们希望轨道交通和家里开的车一样，分很多种。”赵鹏林说，比如有些地铁是“观光线”，不仅沿途的风光非常好，还能凭一张票无数次上下，如同旅游时提供的“通票服务”。再比如，设立可以放大件行李的车厢，今后通过设专门可放大件行李的座位，避免像现在放行李不太方便的现象。“未来地铁初步不仅在干线上铺设，还会在支线、城际线上去建设。”“觉得如果车费不太贵的话，还是愿意考虑的。”昨日市民黄小姐表示，尤其是从老街到机场这一段，老街站每次上下客都很多人，而如果赶上上下班高峰期，特别拥挤，要一路从老街站站到机场，４０、５０分钟还是挺吃力的，宁愿多花点钱也能稍微舒适一点。但是白领林先生则表示，自己每天上下班都要坐地铁，出双倍车资买坐票费用有点高。</content>
```
```python
# -*- coding:utf-8 -*-
from tqdm import tqdm
import jieba

file_raw_path = 'data/corpus_labeled.txt'
label_raw = []
data_raw = []
i = 0
with open(file_raw_path, encoding='utf-8') as fr:
    for line in tqdm(fr):
        if i%2==0:
            label_raw.append(line[5:-6])
        else:
            data_raw.append(line[9:-11])
        i += 1

print('每一个样本有一个url，从中我们可以提取一个话题标签')
[x[:30] for x in label_raw[:len(label_raw):len(label_raw)//10]]
```
```shell
ubuntu:~$ 每一个样本有一个url，从中我们可以提取一个话题标签
          ['http://gongyi.sohu.com/2012070',
           'http://roll.sohu.com/20120625/',
           'http://pic.yule.sohu.com/91258',
           'http://haodf.health.sohu.com/f',
           'http://roll.sohu.com/20120705/',
           'http://db.auto.sohu.com/model_',
           'http://product.it.sohu.com/sea',
           'http://product.it.sohu.com/sea',
           'http://dealer.auto.sohu.com/tj',
           'http://news.sohu.com/20120726/',
           'http://roll.sohu.com/20120706/']
```
```python
print("统计每个类别的文本数量，对数据有一个初步了解")
labels = []
for label in label_raw:
    labels.append(label[7:].split('.')[0])
from collections import Counter
label_stat = Counter(labels)
for k,v in label_stat.most_common(20):
    print('%15s\t\t%d'%(k,v))
```
```shell
ubuntu:~$ 统计每个类别的文本数量，对数据有一个初步了解
             roll		720957
          product		177002
             news		70900
               db		59043
              pic		41236
           sports		38281
            stock		37126
         business		26179
           dealer		25663
              saa		19671
                q		17697
             yule		12779
             drug		11480
            haodf		10890
               it		10797
             data		9110
                s		8678
            money		7448
            daxue		7021
             auto		6843
```
### 根据论文`Character Level Convolutional Neural Networks for Text Classification (2015)`的描述，选择下述5类话题的样本
1. 'sports'
2. 'stock' // finance
3. 'yule'  // entertainment 
4. 'auto'  // automobile
5. 'it'    // technology”
```python
# 定义lambda函数 去掉文本中怪异符号，参考自
# https://gist.github.com/mrvege/2ba6a437f0a4c4812f21#file-filterpunct-py-L5

punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．＊：；Ｏ？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…０１２３４５６７８９''')
## 对str/unicode
filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
## 对list
# filterpuntl = lambda l: list(filter(lambda x: x not in punct, l))

# 选择5个话题，将话题和内容存储
cat_selected = set(['sports', 'stock', 'yule', 'auto', 'it'])
label_selected = []
content_selected = []
for i in tqdm(range(len(labels))):
    if labels[i] in cat_selected and len(data_raw[i])>10:
        label_selected.append(labels[i])
        content_selected.append(filterpunt(data_raw[i]))
        
print('corpus样本\n')
for i in range(0, 5000, 1234):
    print('example %d \n\t%s\n\t%s\n' % (i, label_selected[i], content_selected[i]))
```
```shell
ubuntu:~$ example 0 
          stock
          注每次查询最多显示条

        example 1234 
          sports
          北京时间月日欧洲杯Ｄ组末轮一场比赛在顿涅茨克顿巴斯竞技场展开争夺英格兰客场比力克乌克兰解禁复出的鲁尼打入全场唯一入球英格兰胜平不败战绩获得小组头名决赛将对阵意大利ｓｐｏｒｔｓ全体育图片社Ｖ行峦月日电　据英国每日邮报报道财大气粗的巴黎圣日耳曼队正密谋以万英镑的价格引进切尔西队长约翰特里巴黎圣日耳曼主帅安切洛蒂曾和特里共事一年日前他本想以万英镑的天价引进ＡＣ米兰后卫蒂亚戈席尔瓦但这桩交易在最后一刻宣布告吹豢ㄋ尔财团注资后的巴黎圣日耳曼队并不在乎花钱他们有望以万英镑的价格引进那不勒斯队射手拉维奇而为了得到特里他们准备承诺一份价值万英镑周薪的天价待遇
        ...
```
```python
print("jieba分词，非常费时:\n")
data_words = []
for line in tqdm(content_selected):
    data_words.append([' '.join(jieba.cut(line, cut_all=False))])
    
for i in range(0, 5000, 1234):
    print('sentence %d' % i)
    print(' '.join(data_words[i]))
```
```shell
ubuntu:~$ sentence 0
          注 每次 查询 最 多 显示 条
          sentence 1234
          北京 时间 月 日 欧洲杯 Ｄ 组末 轮 一场 比赛 在 顿涅茨克 顿巴斯 竞技场 展开 争夺 英格兰 客场 比 力克 乌克兰 解禁 复出 的 鲁尼 打入 全场 唯一 入球 英格兰 胜平 不败 战绩 获得 小组 头名 决赛 将 对阵 意大利 ｓ ｐ ｏ ｒ ｔ ｓ 全 体育 图片社 Ｖ 行峦 月 日电 　 据 英国 每日 邮报 报道 财大气粗 的 巴黎 圣日耳曼 队正 密谋 以 万英镑 的 价格 引进 切尔西队 长 约翰 特里 巴黎 圣日耳曼 主帅 安切洛蒂 曾 和 特里 共事 一年 日前 他本 想 以 万英镑 的 天价 引进 Ａ Ｃ 米兰 后卫 蒂 亚戈 席尔瓦 但 这桩 交易 在 最后 一刻 宣布 告吹 豢 ㄋ 尔 财团 注资 后 的 巴黎 圣日耳曼 队 并 不在乎 花钱 他们 有望 以 万英镑 的 价格 引进 那不勒斯 队 射手 拉 维奇 而 为了 得到 特里 他们 准备 承诺 一份 价值 万英镑 周薪 的 天价 待遇
          ...
```
```python
# save train_data & test_data
with open('data/sogou_news_test.txt', 'w') as f:
    for i in range(len(data_words)):
        if i%5==0:
            s = '__label__' + label_selected[i] + ' '
            s = s + " ".join([x for x in data_words[i]])
            f.write(s)
            f.write('\n')

with open('data/sogou_news_train.txt', 'w') as f:
    for i in range(len(data_words)):
        if i%5!=0:
            s = '__label__' + label_selected[i] + ' '
            s = s + " ".join([x for x in data_words[i]])
            f.write(s)
            f.write('\n')
```
```shell
sogou_news_test.txt内容为：
__label__stock 注 每次 查询 最 多 显示 条
__label__stock 实时 行情 编辑 短信 　 Ｇ Ｐ 　 ＋ 　 您 的 股票代码 如 Ｇ Ｐ 　 到 　 　 随时随地 查询 你 的 个股 行情 元 ／ 条不含 通讯费 云南 移动 元 ／ 条 专家 诊股 编辑 短信 　 　 ＋ 　 您 的 股票代码 如 　 到 　 　 　 指明 股票走势 元 ／ 条不含 通讯费 暂 不 支持 福建 ／ 浙江 ／ 湖南 移动用户
...
```
### 加载数据、存储模型；利用模型进行分类
```python
import fasttext

lr = 0.05
dim = 256
classifier = fasttext.supervised(input_file = 'data/sogou_news_train.txt',
                                 output = 'data/intent_model',
                                 label_prefix = '__label__',
                                 dim = dim,
                                 lr = lr,
                                 epoch = 5)
result_tr = classifier.test('data/sogou_news_test.txt')
print(result_tr.precision)
```
```shell
ubuntu:~$ 0.9610818622896665
```

### 加载模型进行分类
```python
# -*- coding:utf-8 -*-
import fasttext

classifier = fasttext.load_model('data/intent_model.bin', label_prefix='__label__')
labels_left = []
texts = []
with open("data/sogou_news_test.txt") as fr:
    for line in fr:
        line = line.rstrip()  # 删除string字符串末尾的指定字符（默认为空格）
        labels_left.append(line.split(" ")[0].replace("__label__",""))  # 正确的标签
        texts.append(line[len(line.split(" ")[0]):])  # 文本内容
labels_predict = [e[0] for e in classifier.predict(texts)]  # 预测输出结果为二维形式
#for e in classifier.predict(texts):
#    print(e)
# 输出为['yule']
        ['it']
        ['auto']
        ...

text_labels = list(set(labels_left))  # 正确的标签（去重）
text_predict_labels = list(set(labels_predict))  # 预测的标签（去重）
print(text_predict_labels)
print(text_labels)

# fromkeys()方法用于创建一个新的字典，其中包含seq的值和设置为value的值
A = dict.fromkeys(text_labels,0)  #预测正确的各个类的数目
B = dict.fromkeys(text_labels,0)   #测试数据集中各个类的数目
C = dict.fromkeys(text_predict_labels,0) #预测结果中各个类的数目
for i in range(0,len(labels_left)):
    B[labels_left[i]] += 1  #测试数据集中各个类的数目+1
    C[labels_predict[i]] += 1  #预测结果中各个类的数目+1
    if labels_left[i] == labels_predict[i]:
        A[labels_left[i]] += 1  #预测正确的各个类的数目+1

print(A) 
print(B)
print(C)
#计算准确率，召回率，F值
for key in B:
    try:
        r = float(A[key]) / float(B[key])  # 召回率（预测正确的 / 实际正确的）
        p = float(A[key]) / float(C[key])  # 准确率（预测正确的 / 预测所有的）
        f = p * r * 2 / (p + r)
        print("%s:\t p:%f\t r:%f\t f:%f" % (key,p,r,f))
    except:
        print("error:", key, "right:", A.get(key,0), "real:", B.get(key,0), "predict:",C.get(key,0))
```
```shell
ubuntu:~$ ['it', 'stock', 'auto', 'yule', 'sports']
          ['it', 'stock', 'auto', 'yule', 'sports']
          {'it': 1660, 'stock': 6173, 'auto': 1043, 'yule': 2365, 'sports': 7379}
          {'it': 1880, 'stock': 6457, 'auto': 1141, 'yule': 2447, 'sports': 7449}
          {'it': 1918, 'stock': 6418, 'auto': 1118, 'yule': 2452, 'sports': 7468}
          it:	    p:0.865485	 r:0.882979	 f:0.874144
          stock:    p:0.961826	 r:0.956017	 f:0.958913
          auto:	    p:0.932916	 r:0.914110	 f:0.923417
          yule:	    p:0.964519	 r:0.966490	 f:0.965503
          sports:   p:0.988082	 r:0.990603	 f:0.989341
```
