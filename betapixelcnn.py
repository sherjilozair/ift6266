import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
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

    @staticmethod
    def fix(x):
        x = x * (1 - 2 * np.finfo(np.float32).eps)
        x = x +  np.finfo(np.float32).eps
        return x

    @staticmethod
    def beta_logpdf(x, alpha_beta):
        x = nn.fix(x)
        alpha_beta = nn.fix(alpha_beta)
        alpha = alpha_beta[..., 0][..., None]
        beta = alpha_beta[..., 1][..., None]
        # negative LL
        logpdf = 0.
        logpdf += (alpha - 1) * tf.log(x)
        logpdf += (beta - 1) * tf.log(1 - x)
        logpdf += tf.lgamma(alpha + beta)
        logpdf -= tf.lgamma(alpha)
        logpdf -= tf.lgamma(beta)
        return - logpdf

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
        self.alpha_beta = nn.conv2d(h, 2, [1, 1], activation_fn=tf.nn.softplus, scope='conv/logits')
        self.losses = nn.beta_logpdf(self.image, self.alpha_beta)
        self.loss = tf.reduce_mean(self.losses)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_params(self, name):
        return self.saver.save(self.sess, name)

    def load_params(self, path):
        self.saver.restore(self.sess, path)

    def train(self, data, mbsz=64, nb_epochs=200):
        for e in xrange(nb_epochs):
            train_losses = []
            for i in xrange(1000):
                image = mnist.train.next_batch(mbsz)[0]
                _, l = self.sess.run([self.train_op, self.loss], {self.image: image})
                train_losses.append(l)
                print 'training...', i, np.mean(train_losses), '\r',

            validation_losses = []
            for j in xrange(100):
                image = mnist.validation.next_batch(mbsz)[0]
                l, = self.sess.run([self.loss], {self.image: image})
                validation_losses.append(l)
                print 'validating...', j, np.mean(validation_losses), '\r',
            path = self.save_params("models/ckpt_{}".format(e))
            self.sample('samples/sample_{}.png'.format(e))
            print "epoch: {}/{}, train loss: {}, validation loss: {}, model saved: {}".format(e, nb_epochs, np.mean(train_losses), np.mean(validation_losses), path)


    def sample(self, name, n=8):
        image = np.zeros((n*n, 28, 28))
        for i in xrange(28):
            for j in xrange(28):
                alpha_beta, = self.sess.run([self.alpha_beta[:, i, j, :]], {self.image: image[..., None]})
                image[:, i, j] = np.random.beta(alpha_beta[..., 0], alpha_beta[..., 1])
                print 'sampling...', i, j, '\r',
        canvas = Image.new('L', (32*n, 32*n))
        for i in xrange(n):
            for j in xrange(n):
                im = Image.fromarray(np.cast[np.uint8](image[i*n+j, :, :] * 255))
                canvas.paste(im, (32*i+2, 32*j+2))
        canvas.save(name)


if __name__ == '__main__':
    model = PixelCNN()
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    model.train(mnist)
