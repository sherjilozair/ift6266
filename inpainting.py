import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio

class Model:
    def __init__(self):
        self.sess = tf.Session()
        self.inputs = h = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.outputs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        for i, rate in enumerate([1, 1, 2, 4, 8]):
            h = slim.conv2d(h, 32, [3, 3], activation_fn=tf.nn.elu,
                padding='SAME', rate=rate, scope='sames/3x3/{}'.format(i))
            h = slim.conv2d(h, 32, [1, 1], activation_fn=tf.nn.elu,
                scope='sames/1x1/{}'.format(i))

        for i, rate in enumerate([8, 4, 2, 1, 1]):
            h = slim.conv2d(h, 32, [3, 3], activation_fn=tf.nn.elu,
                padding='VALID', rate=rate, scope='valids/3x3/{}'.format(i))
            h = slim.conv2d(h, 32, [1, 1], activation_fn=tf.nn.elu,
                scope='valids/1x1/{}'.format(i))

        self.preds = slim.conv2d(h, 3, [1, 1], activation_fn=None, scope='last')
        self.losses = tf.reduce_sum(tf.square(self.outputs - self.preds), axis=[1, 2, 3])
        self.loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def update(self, image):
        image = image / 255.
        inner = image[:, 16:48, 16:48, :]
        image[:, 16:48, 16:48, :] = 0.
        return self.sess.run([self.train_op, self.loss], {self.inputs: image, self.outputs:inner})

# [3, 5, 9, 17] 1 + 2 + 4 + 8

train = np.load('images.train.npz').items()[0][1]
#valid = np.load('images.valid.npz').items()[0]
mbsz = 8
model = Model()

idx = np.arange(train.shape[0])
for e in xrange(10):
    print "epoch", e
    np.random.shuffle(idx)
    for i in xrange(0, train.shape[0], mbsz):
        _, loss = model.update(train[idx[i:i+mbsz]])
        print loss
