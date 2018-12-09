# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 21:57
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : xgboost_model.py
import numpy as np
import tensorflow as tf
from config import valid_ratio, gpu_device, random_seed
from utils import timer


def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i+step])
    return res


def dense(X, size, activation=None, seed=256, l1_scale=0.0):
    he_std = np.sqrt(2 / int(X.shape[1]))
    out = tf.layers.dense(X, units=size, activation=activation,
                          kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=l1_scale),
                          kernel_initializer=tf.random_normal_initializer(stddev=he_std, seed=seed))
    return out


class NNModel:
    def __init__(self, param):
        self.lr = param['lr']
        self.batch_size = param['batch_size']
        self.epoch = param['epoch']
        self.l1_scale = param['l1_scale']
        self.seed = random_seed

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = self.seed
        with graph.device(gpu_device):
            with graph.as_default():
                self.place_x = tf.placeholder(tf.float32, shape=(None, param['input_dim']))
                self.place_y = tf.placeholder(tf.float32, shape=(None, 3, 3))
                self.place_lr = tf.placeholder(tf.float32, shape=(), )

                out = dense(self.place_x, param['dense1'], activation='relu', seed=self.seed, l1_scale=self.l1_scale)
                out = dense(out, param['dense2'], activation='relu', seed=self.seed, l1_scale=self.l1_scale)
                out = dense(out, param['dense3'], activation='relu', seed=self.seed, l1_scale=self.l1_scale)
                out = dense(out, 9, seed=self.seed)
                out = tf.reshape(out, [-1, 3, 3])
                self.out = tf.nn.softmax(out)

                self.loss = tf.losses.absolute_difference(self.place_y, self.out)
                self.reg_loss = self.loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                opt = tf.train.AdamOptimizer(learning_rate=self.place_lr)
                self.train_step = opt.minimize(self.reg_loss)
                init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config, graph=graph)
            self.init = init

    def fit(self, X_train, y_train, is_valid=True):
        if is_valid:
            valid_size = int(X_train.shape[0]*valid_ratio)
            X_train, X_valid = X_train[:-valid_size], X_train[-valid_size:]
            y_train, y_valid = y_train[:-valid_size], y_train[-valid_size:]

        train_idx = np.arange(X_train.shape[0])
        self.session.run(self.init)
        for epoch in range(self.epoch):
            if epoch:
                self.batch_size *= 2
                self.lr /= 4
            with timer(f'fit epoch {epoch}, learning rate: {self.lr:.6f}, batch size: {self.batch_size}'):
                np.random.shuffle(train_idx)
                batches = prepare_batches(train_idx, self.batch_size)
                for rnd, idx in enumerate(batches):
                    feed_dict = {
                        self.place_x: X_train[idx],
                        self.place_y: y_train[idx],
                        self.place_lr: self.lr,
                    }
                    self.session.run(self.train_step, feed_dict=feed_dict)

                    if is_valid and len(batches)//15 > 0 and rnd % (len(batches)//15) == 0:
                        feed_dict = {
                            self.place_x: X_valid,
                            self.place_y: y_valid,
                        }
                        valid_loss, valid_reg_loss = self.session.run([self.loss, self.reg_loss], feed_dict=feed_dict)
                        print(f'rnd: {rnd}, valid_reg_loss: {valid_reg_loss:.4f}, valid_loss: {valid_loss:.4f}')

    def predict(self, X_test):
        return self.session.run(self.out, feed_dict={self.place_x: X_test})

    def save_model(self, path):
        self.saver.save(self.session, path)

    def load_model(self, path):
        self.saver.restore(self.session, path)
