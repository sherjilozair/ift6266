import sys, os
assert len(sys.argv) == 2
expname = sys.argv[1]
home = '/data/lisa/exp/ozairs/ift6266/{}'.format(expname)

import logging
logging.basicConfig(filename='{}.log'.format(expname), level=logging.DEBUG)

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio
from tqdm import tqdm


if not os.path.exists(home):
    os.makedirs(home)
if not os.path.exists(home + '/models/'):
    os.makedirs(home + '/models/')


def residual_block(x, name, sz=5, n_filters=128, padding='SAME'):
    h = x
    h = slim.conv2d(h, n_filters * 2, [1, 1], activation_fn=None,
        scope='{}/1x1/1'.format(name))
    h = slim.conv2d(h, n_filters, [sz, sz], activation_fn=tf.nn.elu,
        padding=padding, scope='{}/szxsz/1'.format(name))
    h = slim.conv2d(h, n_filters, [sz, sz], activation_fn=tf.nn.elu,
        padding=padding, scope='{}/szxsz/2'.format(name))
    h = slim.conv2d(h, n_filters * 2, [1, 1],
        scope='{}/1x1/2'.format(name))
    return x + h

def canvas_block(x, n_filters, scope):
    h = slim.conv2d(x, n_filters * 2, [1, 1], activation_fn=None, scope=scope+'first')
    for i, sz in enumerate([5, 5, 5, 5, 5, 5, 5, 5]):
        h = residual_block(h, scope+str(i), n_filters=n_filters, sz=sz)
    y = slim.conv2d(h, 3, [1, 1], activation_fn=None, scope=scope+'last')
    return y


class Model:
    def __init__(self):
        self.sess = tf.Session()
        self.inputs = h = tf.placeholder(tf.float32, [None, 64, 64, 3])

        y = canvas_block(h, n_filters=128, scope='canvas/')

        self.logits = y[:, 16:48, 16:48, :]

        self.preds = tf.nn.sigmoid(self.logits)

        self.outputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.outputs, logits=self.logits)
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def run(self, images, train=True):
        images = images.copy()
        images = images / 255.
        inner = images[:, 16:48, 16:48, :].copy()
        images[:, 16:48, 16:48, :] = 0.
        fetch = [self.loss] + ([self.train_op] if train else [])
        return self.sess.run(fetch, {self.inputs: images, self.outputs:inner})

    def complete(self, images):
        images = images.copy()
        images = images / 255.
        inner = images[:, 16:48, 16:48, :].copy()
        images[:, 16:48, 16:48, :] = 0.
        predinner, = self.sess.run([self.preds], {self.inputs: images})
        images[:, 16:48, 16:48, :] = predinner
        return images


# [3, 5, 9, 17] 1 + 2 + 4 + 8

datahome = '/data/lisa/exp/ozairs/'
train = np.load(datahome + 'images.train.npz').items()[0][1]
valid = np.load(datahome + 'images.valid.npz').items()[0][1]

mbsz = 16
model = Model()

for e in xrange(200):
    idx = np.arange(train.shape[0])
    np.random.shuffle(idx)
    losses = []
    for i in xrange(0, train.shape[0], mbsz):
        loss, _ = model.run(train[idx[i:i+mbsz]])
        losses.append(loss)
        if i % 100 == 0:
            logging.debug("epoch: {}, iters: {}/{}, train loss: {}".format(e, i, train.shape[0], np.mean(losses)))

    idx = np.arange(valid.shape[0])
    np.random.shuffle(idx)
    losses = []
    for i in xrange(0, valid.shape[0], mbsz):
        loss = model.run(valid[idx[i:i+mbsz]], train=False)
        losses.append(loss)
        if i % 100 == 0:
            logging.debug("epoch: {}, iters: {}/{}, valid loss: {}".format(e, i, valid.shape[0], np.mean(losses)))

    preds = model.complete(train[idx[:mbsz]])
    np.save('{}/train_completions_{}.npy'.format(home, e), preds)

    preds = model.complete(valid[idx[:mbsz]])
    np.save('{}/valid_completions_{}.npy'.format(home, e), preds)

    save_path = model.saver.save(model.sess, '{}/models/'.format(home))
    #print "Saved in {}".format(save_path)
