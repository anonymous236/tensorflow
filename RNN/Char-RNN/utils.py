# -*- coding:utf-8 -*-

import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        # 读入数据文件，一般我们只准备了第一个txt文件，后面再生成后连两个文件
        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    # preprocess data for the first time.
    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            # read() 每次读取整个文件，它通常用于将文件内容放到一个字符串变量中。因此data为一个字符串
            data = f.read()
            
        # 统计一共多少字，相当于用了list(set(data)),
        # collection.Counter这个python模块真是太方便了，很方便的统计次数，推荐！
        counter = collections.Counter(data)
        # 相当于counter.most_commen(),按照次数排序
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # 压缩成一个[(letters),(frequencies)]的形式
        # 得到的char就是(letters)，里面不重复的按照从高到低的顺序存放着每个字符
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        '''
        print(chars)
        >>> (' ', 'e', 't', 'a', 'o', 'n', 'h', 'i', 's', 'r', 'd', 'l', 'u', '\n', 'm', 'w', 'c', 'f', 'y',
             'g', ',', 'p', '.', 'b', '"', 'v', 'k', 'I', "'", 'H', '-', 'T', 'W', 'M', '?', 'S', 'A', 'x',
             'B', 'Y', 'q', 'C', '!', 'N', 'j', 'L', 'O', 'D', 'E', 'P', ';', 'G', 'F', 'z', 'J', 'R', 'V',
             ':', '1', 'U', 'K', '0', '8', '2', '3', '4', 'Q', '7', '5', '6', '9', '£', '[', ']', 'é', '&',
             '(', ')', 'X', 'Z', '`', 'è', 'ñ', 'à', '°', 'ê', '*', 'î', 'ß', '/', 'û', 'ö', 'ü', '½', 'â', '’', 'ô')
        '''
        # vocab是一个字典，将chars中的每个字符进行重新编号
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        '''
        print(vocab)
        >>> {' ': 0, 'e': 1, 't': 2, 'a': 3, 'o': 4, 'n': 5, 'h': 6, 'i': 7, 's': 8, 'r': 9, 'd': 10, 'l': 11,
             'u': 12, '\n': 13, 'm': 14, 'w': 15, 'c': 16, 'f': 17, 'y': 18, 'g': 19, ',': 20, 'p': 21, '.': 22,
             'b': 23, '"': 24, 'v': 25, 'k': 26, 'I': 27, "'": 28, 'H': 29, '-': 30, 'T': 31, 'W': 32, 'M': 33,
             '?': 34, 'S': 35, 'A': 36, 'x': 37, 'B': 38, 'Y': 39, 'q': 40, 'C': 41, '!': 42, 'N': 43, 'j': 44,
             'L': 45, 'O': 46, 'D': 47, 'E': 48, 'P': 49, ';': 50, 'G': 51, 'F': 52, 'z': 53, 'J': 54, 'R': 55,
             'V': 56, ':': 57, '1': 58, 'U': 59, 'K': 60, '0': 61, '8': 62, '2': 63, '3': 64, '4': 65, 'Q': 66,
             '7': 67, '5': 68, '6': 69, '9': 70, '£': 71, '[': 72, ']': 73, 'é': 74, '&': 75, '(': 76, ')': 77,
             'X': 78, 'Z': 79, '`': 80, 'è': 81, 'ñ': 82, 'à': 83, '°': 84, 'ê': 85, '*': 86, 'î': 87, 'ß': 88,
             '/': 89, 'û': 90, 'ö': 91, 'ü': 92, '½': 93, 'â': 94, '’': 95, 'ô': 96}
        '''
        # vocab_file.pkl里存放了一个tuple，里面不重复的按照从高到低的顺序存放着每个字符
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        # data.npy(tensorfile)里存放了每个字符的编号，
        # 这里一共有3381928个字符，那它就是一个长度为3381928的numpy array
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    # 载入变量
    # load the preprocessed the data if the data has been processed before.
    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
    # seperate the whole data into different batches.
    def create_batches(self):
        # 总字符 / (batch大小 * 每个序列长度)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # reshape the original data into the length self.num_batches * self.batch_size * self.seq_length for convenience.
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        # 回想一下rnn的输入和输出，x和y要错一位
        # ydata is the xdata with one position shift.
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        # 将x和y按batch_size切成了很多batch
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        # pointer每个batch过后向后移动一位
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
