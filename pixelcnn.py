import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
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
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.image)
        self.loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2, 3]))
        self.train_op = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss)

    def train(self, data, mbsz=64, nb_epochs=500):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for e in xrange(nb_epochs):
            train_losses = []
            for i in xrange(1000):
                image = mnist.train.next_batch(mbsz)[0]
                _, l = sess.run([self.train_op, self.loss], {self.image: image})
                print 'training', l, '\r'
                train_losses.append(l)

            validation_losses = []
            for j in xrange(100):
                image = mnist.validation.next_batch(mbsz)[0]
                l, = sess.run([self.loss], {self.image: image})
                print 'validating', l, '\r'
                validation_losses.append(l)
            print "epoch: {}/{}, train loss: {}, validation loss: {}".format(e, nb_epochs, np.mean(train_losses), np.mean(validation_losses))



if __name__ == '__main__':
    model = PixelCNN()
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    model.train(mnist)
