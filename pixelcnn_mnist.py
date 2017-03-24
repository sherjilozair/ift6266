import sys, os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

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

    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.labels = tf.cast(self.image, tf.int32)
        h = nn.conv2d(self.image, 256, [5, 5], mask='a', scope='conv/7x7/ul')

        for i in xrange(16):
            ih = h
            h = nn.conv2d(h, 128, [1, 1], scope='conv/1x1/%d/1/ul' % i)
            h = nn.conv2d(h, 128, [3, 3], mask='b', scope='conv/3x3/%d/ul' % i)
            h = nn.conv2d(h, 256, [1, 1], scope='conv/1x1/%d/2/ul' % i)
            h += ih

        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/1')
        h = nn.conv2d(h, 32, [1, 1], scope='conv/relu/1x1/2')
        self.logits = nn.conv2d(h, 256, [1, 1], activation_fn=None, scope='conv/logits')
        self.preds = tf.contrib.distributions.Categorical(self.logits).sample()
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=[1, 2, 3]))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_params(self, name):
        return self.saver.save(self.sess, name)

    def load_params(self, path):
        self.saver.restore(self.sess, path)
        print "params restored from {}".format(path)

    def train(self, name, mnist, mbsz=128, n_epochs=500):

        for e in xrange(n_epochs):
            train_losses = []
            validation_losses = []


            for i in xrange(50000/mbsz):
                image = np.cast[np.float32](mnist.train.next_batch(mbsz)[0]  > 0.5)
                _, l = self.sess.run([self.train_op, self.loss], {self.image: image})
                train_losses.append(l)
                print 'training...', i, np.mean(train_losses), '\r',


            for j in xrange(10000/mbsz):
                image = np.cast[np.float32](mnist.validation.next_batch(mbsz)[0] > 0.5)
                l, = self.sess.run([self.loss], {self.image: image})
                validation_losses.append(l)
                print 'validating...', j, np.mean(validation_losses), '\r',

            path = self.save_params("{}/model".format(name))
            self.sample('{}/sample_{}.png'.format(name, e))
            print "epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e,
                    n_epochs, np.mean(train_losses), np.mean(validation_losses), path)

            del train_losses
            del validation_losses

    def sample(self, name, n=5):
        image = np.zeros((n*n, 28, 28, 1))
        for i in xrange(28):
            for j in xrange(28):
                pixel, = self.sess.run([self.preds[:, i, j, :]], {self.image: image})
                image[:, i, j, :] = pixel
                print 'sampling...', i, j, '\r',

        canvas = Image.new('L', (32*n, 32*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :, 0] * 255))
                canvas.paste(im, (32*i+2, 32*j+2))
        canvas.save(name)


if __name__ == '__main__':
    model = PixelCNN()

    expname = sys.argv[1]
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    if not os.path.exists(expname):
        os.makedirs(expname)
    else:
        ckpt_file = "{}/model".format(expname)
        model.load_params(ckpt_file)

    model.train(expname, mnist)
