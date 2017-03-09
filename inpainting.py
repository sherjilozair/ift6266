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
    h = slim.conv2d(h, n_filters * 2, [1, 1],
        scope='{}/1x1/2'.format(name))
    return x + h


class Model:
    def __init__(self):
        self.sess = tf.Session()
        self.inputs = x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.outputs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        h = slim.conv2d(x, 128, [1, 1], activation_fn=None, scope='first')

        for t in xrange(1):
            for i, rate in enumerate([1, 2, 4, 2, 1, 2, 4, 2, 1]):
                h = block(h, i, n_filters=64, rate=rate)

        y = slim.conv2d(h, 3, [1, 1], activation_fn=None, scope='last')
        self.preds = y[:, 16:48, 16:48, :]

        self.losses = tf.reduce_mean(tf.square(self.outputs - self.preds), axis=[1, 2, 3])
        self.loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def run(self, image, train=True):
        image = image / 255.
        inner = image[:, 16:48, 16:48, :].copy()
        image[:, 16:48, 16:48, :] = 0.
        fetch = [self.loss] + ([self.train_op] if train else [])
        return self.sess.run(fetch, {self.inputs: image, self.outputs:inner})

    def complete(self, images):
        images = images / 255.
        inner = images[:, 16:48, 16:48, :].copy()
        images[:, 16:48, 16:48, :] = 0.
        predinner, = self.sess.run([self.preds], {self.inputs: images})
        print predinner.shape, images.shape
        images[:, 16:48, 16:48, :] = predinner
        return images


# [3, 5, 9, 17] 1 + 2 + 4 + 8

home = '/data/lisa/exp/ozairs/'
train = np.load(home + 'images.train.npz').items()[0][1]
valid = np.load(home + 'images.valid.npz').items()[0][1]

mbsz = 64
model = Model()

for e in xrange(200):
    print "epoch", e
    idx = np.arange(train.shape[0])
    np.random.shuffle(idx)
    losses = []
    for i in xrange(0, train.shape[0], mbsz):
        loss, _ = model.run(train[idx[i:i+mbsz]])
        losses.append(loss)
        print "iters: {}/{}".format(i, train.shape[0]), "train loss", np.mean(losses), '\r',
    print "train loss", np.mean(losses)

    idx = np.arange(valid.shape[0])
    np.random.shuffle(idx)
    losses = []
    for i in xrange(0, valid.shape[0], mbsz):
        loss = model.run(valid[idx[i:i+mbsz]], train=False)
        losses.append(loss)
        print "iters: {}/{}".format(i, valid.shape[0]), "valid loss", np.mean(losses), '\r',
    print "valid loss", np.mean(losses)

    preds = model.complete(valid[idx[:mbsz]])
    np.save('/data/lisa/exp/ozairs/predictions/completions_{}.npy'.format(e), preds)

    save_path = model.saver.save(model.sess, '/data/lisa/exp/ozairs/models/')
    #print "Saved in {}".format(save_path)

