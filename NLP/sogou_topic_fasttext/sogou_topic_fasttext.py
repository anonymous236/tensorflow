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

print("统计每个类别的文本数量，对数据有一个初步了解")
labels = []
for label in label_raw:
    labels.append(label[7:].split('.')[0])
from collections import Counter
label_stat = Counter(labels)
for k,v in label_stat.most_common(20):
    print('%15s\t\t%d'%(k,v))
   
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
    
print("jieba分词，非常费时:\n")
data_words = []
for line in tqdm(content_selected):
    data_words.append([' '.join(jieba.cut(line, cut_all=False))])
    
for i in range(0, 5000, 1234):
    print('sentence %d' % i)
    print(' '.join(data_words[i]))
    
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

'''
ubuntu:~$ 0.9610818622896665
'''
