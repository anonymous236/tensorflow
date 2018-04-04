## py说明
* [intent_fasttext.py](https://github.com/anonymous236/tensorflow/blob/master/NLP/intent_fasttext/intent_fasttext.py) 使用fasttext做意图分类;
  <br>尝试不同的fasttext的learning_rate和feature_dimension，找到最佳的参数
## 使用fasttext做意图分类
意图（intent）是服务类聊天机器人，搜索引擎领域的重要手段，例如下图
![intent](https://github.com/anonymous236/figure/blob/master/Dialogue-Structure-NLP.png)

数据格式为：
```json
{
  "data" : [
    {
      "text" : "i want to go from berlin to tokyo tomorrow",
      "entities" : [
        {
          "entity" : "location",
          "value" : "\"berlin\"",
          "role" : "from",
          "start" : 18,
          "end" : 24
        },
        {
          "entity" : "intent",
          "value" : "\"flight_booking\"",
          "start" : 0,
          "end" : 42
        },
        {
          "entity" : "location",
          "value" : "\"tokyo\"",
          "role" : "to",
          "start" : 28,
          "end" : 33
        },
        {
          "entity" : "datetime",
          "value" : "\"2016-05-29T00:00:00.000-07:00\"",
          "start" : 34,
          "end" : 42
        }
      ]
    }
  ]
}
```
## 代码过程
```python
# -*- coding:utf-8 -*-
import json
import io

# 数据来源：
# 从github下载rasa_nlu项目的repo， 使用`https://github.com/RasaHQ/rasa_nlu/blob/master/test_models/test_model_spacy_sklearn/model_20170628-002705/training_data.json`

# 1. 从json文件读入数据
name = 'data/training_data.json'
with io.open(name, encoding="utf-8-sig") as f:
    data = json.loads(f.read())
    
labels, texts = [], []

# 2. 从json格式的数据提取intent和text
for eg in data['rasa_nlu_data']['common_examples']:
    texts.append(eg['text'])
    labels.append('__label__'  + eg['intent'])

# 3. 将数据分割成 training数据 和 heldout(又名validation)数据
with open('data/intent_small_train.txt', 'w') as f_tr:
    with open('data/intent_small_valid.txt', 'w') as f_val:
        for i in range(len(labels)):
            if i==0 or labels[i]!=labels[i-1]:
                f_val.write(labels[i] + ' ' + texts[i]+'\n')
            else:
                f_tr.write(labels[i] + ' ' + texts[i]+'\n')

# 4. 打印数据，直观了解

print('所有的 intent:')
print(set([x[9:] for x in labels]))
print('\n')

print('所有的 (intent, text) 样本:')
xs = sorted([(labels[i], texts[i]) for i in range(len(labels))])

for i in range(len(labels)):
    print('\t%s : %s' % (xs[i][0][9:], xs[i][1]))
```
```shell
ubuntu:~$ 所有的 intent:
          {'goodbye', 'restaurant_search', 'greet', 'affirm'}

          所有的 (intent, text) 样本:
            affirm : great
            affirm : indeed
            affirm : ok
          ...
```
```python
import fasttext
# fasttext用法的参考文献：https://pypi.python.org/pypi/fasttext

# 我们尝试不同的learning_rate和feature_dimension
lrs = [0.01, 0.05, 0.002]
dims = [5, 10, 25, 50, 75, 100]

best_tr, best_val = 0, 0
for lr in lrs:
    for dim in dims:
        classifier = fasttext.supervised(input_file = 'data/intent_small_train.txt',
                                         output = 'data/intent_model',
                                         label_prefix = '__label__',
                                         dim = dim,
                                         lr = lr,
                                         epoch = 50)
        result_tr = classifier.test('data/intent_small_train.txt')
        result_val = classifier.test('data/intent_small_valid.txt')
        
        if result_tr.precision > best_tr:
            best_tr = result_tr.precision
            params_tr = (lr, dim, result_tr)
            
        if result_val.precision > best_val:
            best_val = result_val.precision
            params_val = (lr, dim, result_val)
print(best_tr)
print(params_tr)
print(best_val)
print(params_val)

classifier = fasttext.supervised(input_file = 'data/intent_small_train.txt',
                                         output = 'data/intent_model',
                                         label_prefix = '__label__',
                                         dim = params_val[1],
                                         lr = params_val[0],
                                         epoch = 50)
print(classifier.predict(['ok ', 'hello', 'bye bye', 'show me chinese restaurants'], k=1))
```
```shell
ubuntu:~$ 0.8918918918918919
          (0.05, 5, <fasttext.model.ClassifierTestResult object at 0x7fe0876fff98>)
          0.8571428571428571
          (0.05, 5, <fasttext.model.ClassifierTestResult object at 0x7fe0876841d0>)
          [['affirm'], ['greet'], ['goodbye'], ['restaurant_search']]
```
