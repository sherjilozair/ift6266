import sys, os
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
            mask_filter = np.ones(shape, dtype=np.bool)
            mask_filter[mid_x, mid_y+1:, :, :] = False
            mask_filter[mid_x+1:, :, :, :] = False

            if mask == 'a':
                mask_filter[mid_x, mid_y, :, :] = False

            W  = W * mask_filter

        h = tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)

        if activation_fn:
            h = activation_fn(h)

        return h


class PixelCNN:
    mask = np.ones((64, 64, 1), dtype=np.bool)
    mask[16:48, 16:48, :] = False

    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.outer = self.image * self.mask
        h = nn.conv2d(self.image, 256, [7, 7], mask='a', activation_fn=None, scope='conv/7x7/ul')
        j = nn.conv2d(self.outer, 256, [7, 7], activation_fn=None, scope='conv/7x7/ff')

        for i in xrange(8):
            ih = h
            h = nn.conv2d(h, 128, [1, 1], scope='conv/1x1/%d/1/ul' % i)
            h = nn.conv2d(h, 128, [5, 5], mask='b', scope='conv/3x3/%d/ul' % i)
            h = nn.conv2d(h, 256, [1, 1], activation_fn=None, scope='conv/1x1/%d/2/ul' % i)
            h += ih

            ij = j
            j = nn.conv2d(j, 128, [1, 1], scope='conv/1x1/%d/1/ff' % i)
            j = nn.conv2d(j, 128, [5, 5], scope='conv/3x3/%d/ff' % i)
            j = nn.conv2d(j, 256, [1, 1], scope='conv/1x1/%d/2/ff' % i)
            j += ij

            h = nn.conv2d(tf.concat([h, j], axis=3), 256, [1, 1], activation_fn=None, scope='conv/concat/%d' % i)

        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/1')
        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/2')
        self.logits = nn.conv2d(h, 3, [1, 1], activation_fn=None, scope='conv/logits')

        self.preds = tf.nn.sigmoid(self.logits)
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.image)
        self.losses = self.losses[:, 16:48, 16:48, :]
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_params(self, name):
        return self.saver.save(self.sess, name)

    def load_params(self, path):
        self.saver.restore(self.sess, path)
        print "params restored from {}".format(path)

    def train(self, name, train, valid, mbsz=32, nb_epochs=200):
        tidx = np.arange(len(train))
        vidx = np.arange(len(valid))

        for e in xrange(nb_epochs):
            train_losses = []
            validation_losses = []

            np.random.shuffle(tidx)

            for i in xrange(0, len(tidx)/5, mbsz):
                image = train[tidx[i:i+mbsz]]
                _, l = self.sess.run([self.train_op, self.loss], {self.image: image})
                train_losses.append(l)
                print 'training...', i, np.mean(train_losses), '\r',

            np.random.shuffle(vidx)

            for j in xrange(0, len(vidx)/5, mbsz):
                image = valid[vidx[i:i+mbsz]]
                l, = self.sess.run([self.loss], {self.image: image})
                validation_losses.append(l)
                print 'validating...', j, np.mean(validation_losses), '\r',

            path = self.save_params("{}/model".format(name))
            self.sample('{}/sample_{}.png'.format(name, e), train[tidx[:16]])
            print "epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e,
                    nb_epochs, np.mean(train_losses), np.mean(validation_losses), path)

            del train_losses
            del validation_losses

    def sample(self, name, image, n=4):
        image[:, 16:48, 16:48, :] = 0.
        for i in xrange(16, 48):
            for j in xrange(16, 48):
                pixel, = self.sess.run([self.preds[:, i, j, :]], {self.image: image})
                image[:, i, j, :] = pixel
                print 'sampling...', i, j, '\r',

        canvas = Image.new('RGB', (72*n, 72*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, :] * 255))
                canvas.paste(im, (72*i+4, 72*j+4))
        canvas.save(name)


if __name__ == '__main__':
    model = PixelCNN()

    expname = sys.argv[1]
    datahome = '/data/lisa/exp/ozairs/'
    train = np.load(datahome + 'images.train.npz').items()[0][1] / 255.
    valid = np.load(datahome + 'images.valid.npz').items()[0][1] / 255.

    if not os.path.exists(expname):
        os.makedirs(expname)
    else:
        ckpt_file = "{}/model".format(expname)
        model.load_params(ckpt_file)

    model.train(expname, train, valid)
