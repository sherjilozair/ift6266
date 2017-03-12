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


def block(x, name, rate=1, n_filters=32, padding='SAME'):
    h = x
    h = slim.conv2d(h, n_filters * 2, [1, 1], activation_fn=None,
        scope='{}/1x1/1'.format(name))
    h = slim.conv2d(h, n_filters, [3, 3], activation_fn=tf.nn.elu,
        padding=padding, rate=rate, scope='{}/3x3/1'.format(name))
    h = slim.conv2d(h, n_filters, [3, 3], activation_fn=tf.nn.elu,
        padding=padding, rate=rate, scope='{}/3x3/2'.format(name))
    h = slim.conv2d(h, n_filters * 2, [1, 1],
        scope='{}/1x1/2'.format(name))
    return x + h


class Model:
    def __init__(self, n_filters=64):
        self.sess = tf.Session()
        self.image = x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        m = np.zeros((mbsz, 64, 64, 3))
        m[:, 16:48, 16:48, :] = 1.
        self.mask = m
        self.inputs = x * (1 - m) # 0s outside, 1s inside # no randomness

        with tf.name_scope('gen') as scope:
            h = slim.conv2d(self.inputs, 2*n_filters, [1, 1], activation_fn=None, scope=scope+'first')
            for i, rate in enumerate([1, 2, 4, 2, 1, 2, 4, 2, 1]):
                h = block(h, scope+str(i), n_filters=n_filters, rate=rate)
            y = slim.conv2d(h, 3, [1, 1], activation_fn=tf.nn.sigmoid, scope=scope+'last')

        # add predicted inner to actual outer
        self.g = h = y * m + x * (1 - m)

        def discriminator(h):
            with tf.name_scope('disc') as scope:
                with slim.arg_scope([slim.conv2d], padding='VALID'):
                    h = slim.conv2d(h, 48, [3, 3], scope=scope+'dconv1')
                    h = slim.conv2d(h, 48, [3, 3], scope=scope+'dconv2')
                    h = slim.conv2d(h, 48, [3, 3], stride=[2, 2], scope=scope+'dconv3')

                    h = slim.conv2d(h, 96, [3, 3], scope=scope+'dconv4')
                    h = slim.conv2d(h, 96, [3, 3], scope=scope+'dconv5')
                    h = slim.conv2d(h, 96, [3, 3], stride=[2, 2], scope=scope+'dconv6')

                    h = slim.conv2d(h, 192, [3, 3], scope=scope+'dconv7')
                    h = slim.conv2d(h, 192, [3, 3], scope=scope+'dconv8')
                    h = slim.conv2d(h, 192, [3, 3], stride=[2, 2], scope=scope+'dconv9')

                    h = slim.conv2d(h, 192, [3, 3], scope=scope+'dconv10')
                    h = slim.conv2d(h, 48, [1, 1], scope=scope+'dconv11')
                    h = slim.conv2d(h, 1, [1, 1], activation_fn=None, scope=scope+'dconv12')
                    d = slim.flatten(h)
                    return d

        d_x = discriminator(self.image) # real
        d_g = discriminator(self.g) # generated

        # l_d_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_x), logits=d_x))
        # l_d_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g), logits=d_g))

        self.dloss = tf.reduce_mean(d_g - d_x)
        self.gloss = tf.reduce_mean(- d_g)

        self.traind_op = tf.train.AdamOptimizer(1e-4).minimize(self.dloss, var_list=slim.get_variables(scope="disc"))
        self.traing_op = tf.train.AdamOptimizer(1e-4).minimize(self.gloss, var_list=slim.get_variables(scope="gen"))

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def rung(self, image, train=True):
        image = image.copy() / 255.
        fetch = [self.gloss] + ([self.traing_op] if train else [])
        return self.sess.run(fetch, {self.image: image})

    def rund(self, image, train=True):
        image = image.copy() / 255.
        fetch = [self.dloss] + ([self.traind_op] if train else [])
        return self.sess.run(fetch, {self.image: image})

    def complete(self, image):
        image = image.copy() / 255.
        return self.sess.run(self.g, {self.image: image})


datahome = '/data/lisa/exp/ozairs/'
train = np.load(datahome + 'images.train.npz').items()[0][1]
valid = np.load(datahome + 'images.valid.npz').items()[0][1]

mbsz = 64
model = Model()

for e in xrange(200):
    logging.debug("epoch: {}".format(e))
    idx = np.arange(train.shape[0])
    np.random.shuffle(idx)
    losses = []
    for i in xrange(0, train.shape[0], mbsz):
        gloss, _ = model.rung(train[idx[i:i+mbsz]])
        dloss, _ = model.rund(train[idx[i:i+mbsz]])
        losses.append((dloss, gloss))
        logging.debug("iters: {}/{}, train loss: {}".format(i, train.shape[0], np.mean(losses, axis=0)))

    preds = model.complete(train[idx[:mbsz]])
    np.save('{}/train_completions_{}.npy'.format(home, e), preds)

    preds = model.complete(valid[idx[:mbsz]])
    np.save('{}/valid_completions_{}.npy'.format(home, e), preds)

    save_path = model.saver.save(model.sess, '{}/models/'.format(home))
    #print "Saved in {}".format(save_path)

