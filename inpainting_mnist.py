import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio
from tqdm import tqdm
import sys, os
assert len(sys.argv) == 2
expname = sys.argv[1]
home = '/data/lisa/exp/ozairs/ift6266/{}'.format(expname)

if not os.path.exists(home):
    os.makedirs(home)

def block(x, name, rate=1, n_filters=32, padding='SAME'):
    h = x
    h = slim.conv2d(h, n_filters * 2, [1, 1], activation_fn=None,
        scope='{}/1x1/1'.format(name))
    h = slim.conv2d(h, n_filters, [3, 3], activation_fn=tf.nn.elu,
        padding=padding, rate=rate, scope='{}/3x3/1'.format(name))
    h = slim.conv2d(h, n_filters, [3, 3], activation_fn=tf.nn.elu,
        padding=padding, rate=rate, scope='{}/3x3/2'.format(name))
    h = slim.conv2d(h, n_filters * 2, [1, 1], activation_fn=None,
        scope='{}/1x1/2'.format(name))
    return x + h


class Model:
    def __init__(self):
        self.sess = tf.Session()
        self.inputs = x = tf.placeholder(tf.float32, [None, 28, 28, 1])

        h = slim.conv2d(x, 128, [1, 1], activation_fn=None, scope='first')

        for t in xrange(1):
            for i, rate in enumerate([1, 2, 4, 2, 1, 2, 4, 2, 1]):
                h = block(h, i, n_filters=64, rate=rate)

        y = slim.conv2d(h, 1, [1, 1], activation_fn=None, scope='last')
        self.logits = y[:, 7:21, 7:21, :]
        self.preds = tf.nn.sigmoid(self.logits)

        self.outputs = tf.placeholder(tf.float32, [None, 14, 14, 1])

        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.outputs, logits=self.logits)
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def run(self, image, train=True):
        inner = image[:, 7:21, 7:21, :].copy()
        image[:, 7:21, 7:21, :] = 0.
        fetch = [self.loss] + ([self.train_op] if train else [])
        return self.sess.run(fetch, {self.inputs: image, self.outputs:inner})

    def complete(self, image):
        inner = image[:, 7:21, 7:21, :].copy()
        image[:, 7:21, 7:21, :] = 0.
        predinner, = self.sess.run([self.preds], {self.inputs: image})
        image[:, 7:21, 7:21, :] = predinner
        return image

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mbsz = 64
model = Model()

losses = []

for i in xrange(10**6):
    x = mnist.train.next_batch(mbsz)[0].reshape(-1, 28, 28, 1).copy()
    loss, _ = model.run(x)
    losses.append(loss)
    mloss = np.mean(losses[-1000:])
    print i, mloss, '\r',


    if i % 1000 == 0:
        print i, mloss
        preds = model.complete(mnist.train.next_batch(mbsz)[0].reshape(-1, 28, 28, 1))
        trainloc = '{}/train_completions_{}.npy'.format(home, i)
        np.save(trainloc, preds)
        preds = model.complete(mnist.validation.next_batch(mbsz)[0].reshape(-1, 28, 28, 1))
        validloc = '{}/valid_completions_{}.npy'.format(home, i)
        np.save(validloc, preds)
        save_path = model.saver.save(model.sess, '/data/lisa/exp/ozairs/mnist_models/')
        print trainloc
        print validloc
        print save_path
