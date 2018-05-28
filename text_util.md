## 清理数据

* https://gist.github.com/mrvege/2ba6a437f0a4c4812f21#file-filterpunct-py-L5
```python
#!/usr/bin/env python
# encoding: utf-8
__author__ = 'dm'

punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
# 对str/unicode
filterpunt = lambda s : ''.join(filter(lambda x : x not in punct , s))
# 对list
filterpuntl = lambda l : list(filter(lambda x : x not in punct , l))
```

* 设计清理中文字符
```python
## 分词:
for line in tqdm(text_lines):
    one_line = [' '.join(jieba.cut(line, cut_all=False))][0].split(' ')
    data_words.extend(one_line)
    data_lines.append(one_line)
    
# 去掉符号和数字
import re
# 标点符号 (punctuation)
punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．＊：；Ｏ？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…０１２３４５６７８９''')
isNumber = re.compile(r'\d+.*')

filter_words = [w for w in data_words if (w not in punct)
                and (not isNumber.search(w.lower()))]
```

* https://github.com/dapurv5/neural_kbqa/blob/master/code/movieqa/text_util.py
```python
#!/usr/bin/python

"""
Utilities for cleaning the text data
"""
import unicodedata

def clean_word(word):
    word = word.strip('\n')
    word = word.strip('\r')
    word = word.lower()
    word = word.replace('%', '') #99 and 44/100% dead
    word = word.strip()
    word = word.replace(',', '')
    word = word.replace('.', '')
    word = word.replace('"', '')
    word = word.replace('\'', '')
    word = word.replace('?', '')
    word = word.replace('|', '')
    word = unicode(word, "utf-8") #Convert str -> unicode (Remember default encoding is ascii in python)
    word = unicodedata.normalize('NFKD', word).encode('ascii','ignore') #Convert normalized unicode to python str
    word = word.lower() #Don't remove this line, lowercase after the unicode normalization
    return word


def clean_line(line):
    """
    Do not replace PIPE here.
    """
    line = line.strip('\n')
    line = line.strip('\r')
    line = line.strip()
    line = line.lower()
    return line

def append_word_to_str(text, str):
    if len(text) == 0:
        return str
    else:
        return text + " " + str

if __name__ == "__main__":
    print "__"+clean_word("  ")+"__"
```
