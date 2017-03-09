import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio
from tqdm import tqdm

def block(x, name, rate=1, n_filters=32, padding='SAME'):
    h = x
    h = slim.conv2d(h, n_filters * 2, [1, 1], activation_fn=None,
        scope='{}/1x1/1'.format(name))
    h = slim.conv2d(h, n_filters, [3, 3], activation_fn=tf.nn.elu,
        padding=padding, rate=rate, scope='{}/3x3/1'.format(name))
    h = slim.conv2d(h, n_filters, [3, 3], activation_fn=tf.nn.elu,
        padding=padding, rate=rate, scope='{}/3x3/2'.format(name))
    h = slim.conv2d(h, n_filters * 2, [1, 1], activation_fn=None
        scope='{}/1x1/2'.format(name))
    return x + h


class Model:
    def __init__(self):
        self.sess = tf.Session()
        self.inputs = x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.outputs = tf.placeholder(tf.float32, [None, 14, 14, 1])

        h = slim.conv2d(x, 128, [1, 1], activation_fn=None, scope='first')

        for t in xrange(1):
            for i, rate in enumerate([1, 2, 1, 2, 1, 2, 1]):
                h = block(h, i, n_filters=64, rate=rate)

        y = slim.conv2d(h, 1, [1, 1], activation_fn=None, scope='last')
        self.preds = y[:, 7:21, 7:21, :]

        self.losses = tf.reduce_sum(tf.square(self.outputs - self.preds), axis=[1, 2, 3])
        self.loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss)

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

tlosses = []
vlosses = []

for i in xrange(10**6):
    tx = mnist.train.next_batch(mbsz)[0].reshape(-1, 28, 28, 1)
    loss, _ = model.run(tx)
    tlosses.append(loss)

    vx = mnist.validation.next_batch(mbsz)[0].reshape(-1, 28, 28, 1)
    loss, = model.run(vx, train=False)
    vlosses.append(loss)

    print i, np.mean(tlosses[-1000:]), np.mean(vlosses[-1000:])#, '\r',

    if i % 1000 == 0:
        preds = model.complete(mnist.train.next_batch(mbsz)[0].reshape(-1, 28, 28, 1))
        np.save('/data/lisa/exp/ozairs/predictions/mnist_train_completions_{}.npy'.format(i), preds)
        preds = model.complete(mnist.validation.next_batch(mbsz)[0].reshape(-1, 28, 28, 1))
        np.save('/data/lisa/exp/ozairs/predictions/mnist_valid_completions_{}.npy'.format(i), preds)

    #save_path = model.saver.save(model.sess, '/data/lisa/exp/ozairs/mnist_models/')
    #print "Saved in {}".format(save_path)

