import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

class nn:

    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size, mask=None, activation_fn=tf.nn.relu, scope=None):
        shape = kernel_size + [inputs.get_shape()[-1]] + [num_outputs]
        weights_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(scope):
            W = tf.get_variable('W', shape, tf.float32, weights_initializer)
            b = tf.get_variable('b', num_outputs, tf.float32, tf.zeros_initializer())

        if mask:
            mid_x = shape[0]/2
            mid_y = shape[1]/2
            mask_filter = np.ones(shape, dtype=np.float32)
            mask_filter[mid_x, mid_y+1:, :, :] = 0.
            mask_filter[mid_x+1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[mid_x, mid_y, :, :] = 0.

            W *= mask_filter

        h = tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)

        if activation_fn:
            h = activation_fn(h)

        return h


class PixelCNN:
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 28, 28, 1])
        h = nn.conv2d(self.image, 256, [7, 7], mask='a', scope='conv/7x7')
        for i in xrange(12):
            ih = h
            h = nn.conv2d(h, 128, [1, 1], scope='conv/1x1/%d/1' % i)
            h = nn.conv2d(h, 128, [3, 3], mask='b', scope='conv/3x3/%d' % i)
            h = nn.conv2d(h, 256, [1, 1], scope='conv/1x1/%d/2' % i)
            h = ih + h
        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/1')
        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/2')
        self.logits = nn.conv2d(h, 1, [1, 1], activation_fn=None, scope='conv/logits')
        self.means = tf.nn.sigmoid(self.logits)
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.image)
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_params(self, name):
        return self.saver.save(self.sess, name)

    def load_params(self, path):
        self.saver.restore(self.sess, path)

    def train(self, train, valid, mbsz=64, nb_epochs=200):
        for e in xrange(nb_epochs):
            train_losses = []
            idx = np.arange(len(train))
            np.random.shuffle(idx)
            for i in xrange(1000):
                image = train[idx[mbsz*i:mbsz*(i+1)]]
                _, l = self.sess.run([self.train_op, self.loss], {self.image: image})
                train_losses.append(l)
                print 'training...', i, np.mean(train_losses), '\r',

            validation_losses = []
            idx = np.arange(len(valid))
            np.random.shuffle(idx)

            for j in xrange(100):
                image = valid[idx[mbsz*i:mbsz*(i+1)]]
                l, = self.sess.run([self.loss], {self.image: image})
                validation_losses.append(l)
                print 'validating...', j, np.mean(validation_losses), '\r',

            path = self.save_params("models/ckpt_{}".format(e))
            #self.sample('samples/sample_{}.png'.format(e))
            print "epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e, nb_epochs, np.mean(train_losses), np.mean(validation_losses), path)


    def sample(self, name, n=8):
        means = np.zeros((n*n, 28, 28, 1))
        image = np.zeros((n*n, 28, 28, 1))
        for i in xrange(28):
            for j in xrange(28):
                pixelmean, = self.sess.run([self.means[:, i, j, :]], {self.image: image})
                means[:, i, j, :] = pixelmean
                image[:, i, j, :] = np.cast[np.float32](np.random.random(pixelmean.shape) < pixelmean)
                print 'sampling...', i, j, '\r',
        canvas = Image.new('L', (32*n, 32*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](means[i*n+j, :, :, 0] * 255))
                canvas.paste(im, (32*i+2, 32*j+2))
        canvas.save(name)


if __name__ == '__main__':
    model = PixelCNN()

    datahome = '/data/lisa/exp/ozairs/'
    train = np.load(datahome + 'images.train.npz').items()[0][1]
    valid = np.load(datahome + 'images.valid.npz').items()[0][1]

    model.train(train, valid)
