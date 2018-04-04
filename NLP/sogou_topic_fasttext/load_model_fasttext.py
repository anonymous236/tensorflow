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
        
'''
['it', 'stock', 'auto', 'yule', 'sports']
['it', 'stock', 'auto', 'yule', 'sports']
{'it': 1660, 'stock': 6173, 'auto': 1043, 'yule': 2365, 'sports': 7379}
{'it': 1880, 'stock': 6457, 'auto': 1141, 'yule': 2447, 'sports': 7449}
{'it': 1918, 'stock': 6418, 'auto': 1118, 'yule': 2452, 'sports': 7468}
it:	    p:0.865485	 r:0.882979	 f:0.874144
stock:    p:0.961826	 r:0.956017	 f:0.958913
auto:	    p:0.932916	 r:0.914110	 f:0.923417
yule:	    p:0.964519	 r:0.966490	 f:0.965503
sports:   p:0.988082	 r:0.990603	 f:0.989341
'''
