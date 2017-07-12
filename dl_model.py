from abstract_model import AbstractModel
import tensorlayer as tl
import tensorflow as tf
import data_helper
import numpy as np

class SimpleDNN(AbstractModel):
    sess = None

    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, 520])
        self.y = tf.placeholder(tf.float32, [None, 4])
        network = tl.layers.InputLayer(self.x, name='Input')
        network = tl.layers.DenseLayer(network, n_units = 128, act = tf.nn.relu, name='fc1')
        network = tl.layers.DenseLayer(network, n_units = 128, act = tf.nn.relu, name='fc2')
        network = tl.layers.DenseLayer(network, n_units = 64, act = tf.nn.relu, name='fc3')
        network = tl.layers.DenseLayer(network, n_units = 32, act = tf.nn.relu, name='fc4')
        network = tl.layers.DenseLayer(network, n_units = 16, act = tf.nn.relu, name='fc5')
        network = tl.layers.DenseLayer(network, n_units = 8, act = tf.nn.relu, name='fc6')
        network = tl.layers.DenseLayer(network, n_units = 4, act = tf.nn.relu, name='fc7')
        self.y_ = network.outputs

        self.cost = tl.cost.mean_squared_error(self.y, self.y_)
        self.optimize = tf.train.AdamOptimizer().minimize(self.cost)

    def fit(self, x, y, epoch=1000000, batch_size=256):
        # Data pre-processing
        self._preprocess(x, y)
        y = np.concatenate((
            np.expand_dims(self.longitude_normalize_y, -1), np.expand_dims(self.latitude_normalize_y, -1)
        ), axis=-1)
        y = np.concatenate((y, np.expand_dims(self.floor_y, -1)), axis=-1)
        y = np.concatenate((y, np.expand_dims(self.buildingID_y, -1)), axis=-1)

        # Train the model
        self.sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            mini_x = data_helper.getMiniBatch(self.normalize_x, batch_size)
            mini_y = data_helper.getMiniBatch(y, batch_size)
            feed_dict = {
                self.x: mini_x.next(),
                self.y: mini_y.next()
            }
            _cost, _ = self.sess.run([self.cost, self.optimize], feed_dict=feed_dict)
            if i % 100 == 0:
                print "epoch: ", i, '\tcost: ', _cost
        self.save()

    def save(self, save_path='./simple_dnn.ckpt'):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

class ComplexDNN(AbstractModel):
    sess = None
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, 520])
        self.y = tf.placeholder(tf.float32, [None, 2])
        network = tl.layers.InputLayer(self.x, name='Input')
        network = tl.layers.DenseLayer(network, n_units = 1024, act = tf.nn.relu, name='fc1')
        network = tl.layers.DenseLayer(network, n_units = 512, act = tf.nn.relu, name='fc2')
        network = tl.layers.DenseLayer(network, n_units = 256, act = tf.nn.relu, name='fc3')
        network = tl.layers.DenseLayer(network, n_units = 64, act = tf.nn.relu, name='fc4')
        network = tl.layers.DenseLayer(network, n_units = 32, act = tf.nn.relu, name='fc5')
        network = tl.layers.DenseLayer(network, n_units = 2, act = tf.nn.relu, name='fc6')
        self.y_ = network.outputs

        self.cost = tl.cost.mean_squared_error(self.y, self.y_)
        self.optimize = tf.train.AdamOptimizer().minimize(self.cost)

    def fit(self, x, y, epoch=1000, batch_size=1024):
        # Data pre-processing
        self._preprocess(x, y)
        y_pair = np.concatenate((
            np.expand_dims(self.longitude_normalize_y, -1), np.expand_dims(self.latitude_normalize_y, -1)
        ), axis=-1)

        # Train the model
        print "<< training >>"
        self.sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            mini_x = data_helper.getMiniBatch(self.normalize_x , batch_size)
            mini_y = data_helper.getMiniBatch(y_pair, batch_size)
            feed_dict = {
                self.x: mini_x.next(),
                self.y: mini_y.next()
            }
            _cost, _, _output = self.sess.run([self.cost, self.optimize, self.y_], feed_dict=feed_dict)
            if i % 100 == 0:
                print "epoch: ", i, '\tcost: ', _cost
        print 'y : ', feed_dict[self.y][:5]
        print 'y_: ', _output[:5]
        self.save()

    def save(self, save_path='./complex_dnn.ckpt'):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
