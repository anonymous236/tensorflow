# -*- coding:utf-8 -*-

from __future__ import print_function

import argparse
import time
import os
from six.moves import cPickle

# 命令行参数选项
# 每个参数的"help"说明里都有了详细的解释
parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
# 输入数据路径
parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                    help='data directory containing input.txt with training examples')
# 存储模型的路径
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory to store tensorboard logs')
# 保存模型的频率
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save frequency. Number of passes between checkpoints of the model.')
# 训练模型时的起始文件
parser.add_argument('--init_from', type=str, default=None,
                    help="""continue training from saved model at this path (usually "save").
                        Path must contain files saved by previous training process:
                        'config.pkl'        : configuration;
                        'chars_vocab.pkl'   : vocabulary definitions;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                         Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
                    """)
# Model params
# rnn类型
parser.add_argument('--model', type=str, default='lstm',
                    help='lstm, rnn, gru, or nas')
# rnn的cell内神经元数目
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
# rnn层数
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
# Optimization
# 每个序列的长度
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN sequence length. Number of timesteps to unroll for.')
# batch size
parser.add_argument('--batch_size', type=int, default=50,
                    help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
# epoch数目
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs. Number of full passes through the training examples.')
# 梯度clip（防止梯度爆炸）
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='clip gradients at this value')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
# 学习率削减时用到的参数
parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='decay rate for rmsprop')
parser.add_argument('--output_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the hidden layer')
parser.add_argument('--input_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the input layer')
args = parser.parse_args()

import tensorflow as tf
from utils import TextLoader
from model import Model

def train(args):
    # 加载数据，解释详见util.py文件
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    # 如果要从原来的模型基础上继续训练的话，执行这一程序块
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.latest_checkpoint(args.init_from)
        assert ckpt, "No checkpoint found"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        
    # 这里的.pkl文件我们在model.py里面保存模型的时候还会看到～
    # 这里就是把以前保存的模型又调了出来
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    # 我们的模型文件，类似caffe里面的train_net.prototxt文件
    # 前面的参数传递进去，具体解释详见model.py
    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        # 所有变量初始化并运行
        sess.run(tf.global_variables_initializer())
        # 创建一个saver，便于后面的模型保存和重载
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        # 加载模型
        if args.init_from is not None:
            saver.restore(sess, ckpt)
        # e代表每个epoch
        for e in range(args.num_epochs):
            # tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number, 也就是重新赋值
            # 学习率的dacay，lr = learning_rate * decay_rate^e
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            # 状态都初始化
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            # 提出不同的batch然后feed进model
            # b代表每个batch
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                # instrument for tensorboard
                # 运行模型，跑出来一个结果
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))
                # 当到达保存步数或训练到最后一步时保存模型
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    train(args)
