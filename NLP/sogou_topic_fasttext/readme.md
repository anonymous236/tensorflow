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
```
```python
# save model
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
