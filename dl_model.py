import tensorlayer as tl
import tensorflow as tf
import data_helper

class SimpleDNN(object):
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

    def fit(self, x, y, epoch=200000, batch_size=256):
        self.sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            mini_x = data_helper.getMiniBatch(x, batch_size)
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