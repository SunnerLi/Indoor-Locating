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
        network = tl.layers.DenseLayer(network, n_units = 256, act = tf.nn.relu, name='fc1')
        network = tl.layers.DenseLayer(network, n_units = 128, act = tf.nn.relu, name='fc2')
        network = tl.layers.DenseLayer(network, n_units = 128, act = tf.nn.relu, name='fc3')
        network = tl.layers.DenseLayer(network, n_units =  64, act = tf.nn.relu, name='fc4')
        network = tl.layers.DenseLayer(network, n_units =  64, act = tf.nn.relu, name='fc5')
        network = tl.layers.DenseLayer(network, n_units =  32, act = tf.nn.relu, name='fc6')
        network = tl.layers.DenseLayer(network, n_units =  32, act = tf.nn.relu, name='fc7')
        network = tl.layers.DenseLayer(network, n_units =  16, act = tf.nn.relu, name='fc8')
        network = tl.layers.DenseLayer(network, n_units =   8, act = tf.nn.relu, name='fc9')
        network = tl.layers.DenseLayer(network, n_units =   4, act = tf.nn.relu, name='fc10')
        self.y_ = network.outputs

        self.cost = tl.cost.mean_squared_error(self.y, self.y_)
        self.optimize = tf.train.AdamOptimizer().minimize(self.cost)

    def fit(self, x, y, epoch=10000, batch_size=256):
        # Data pre-processing
        self._preprocess(x, y)
        y = np.concatenate((
            np.expand_dims(self.longitude_normalize_y, -1), np.expand_dims(self.latitude_normalize_y, -1)
        ), axis=-1)
        y = np.concatenate((y, np.expand_dims(self.floorID_y, -1)), axis=-1)
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
    # Model save path
    longitude_regression_model_save_path = './complexDnn_long.pkl'
    latitude_regression_model_save_path = './complexDnn_lat.pkl'
    floor_classifier_save_path = './complexDnn_floor.pkl'
    building_classifier_save_path = './complexDnn_building.pkl'

    sess = None
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, 520])
        self.locating_y = tf.placeholder(tf.float32, [None, 2])
        self.building_y = tf.placeholder(tf.float32, [None, 1])
        self.floor_y = tf.placeholder(tf.float32, [None, 1])
        self.alternative_ctl = tf.placeholder(tf.bool)

        locating_network = tl.layers.InputLayer(self.x, name='Input')
        locating_network = tl.layers.DenseLayer(locating_network, n_units = 256, act = tf.nn.relu, name='locating_fc1' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units = 128, act = tf.nn.relu, name='locating_fc2' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units = 128, act = tf.nn.relu, name='locating_fc3' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units =  64, act = tf.nn.relu, name='locating_fc4' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units =  64, act = tf.nn.relu, name='locating_fc5' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units =  32, act = tf.nn.relu, name='locating_fc6' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units =  32, act = tf.identity, name='locating_fc7' )
        locating_network = tl.layers.DenseLayer(locating_network, n_units =   2, act = tf.identity, name='locating_fc8' )
        self.locating_predict_y = locating_network.outputs
        self.locating_cost = tl.cost.mean_squared_error(self.locating_y, self.locating_predict_y)
        self.locating_optimize = tf.train.AdamOptimizer().minimize(self.locating_cost)

        building_network = tl.layers.InputLayer(self.locating_predict_y, name='building_input')
        building_network = tl.layers.DenseLayer(building_network, n_units = 128, act = tf.identity, name='building_fc1')
        building_network = tl.layers.DenseLayer(building_network, n_units =  32, act = tf.identity, name='building_fc2')
        building_network = tl.layers.DenseLayer(building_network, n_units =  16, act = tf.nn.relu, name='building_fc3')
        building_network = tl.layers.DenseLayer(building_network, n_units =   8, act = tf.nn.relu, name='building_fc4')
        building_network = tl.layers.DenseLayer(building_network, n_units =   4, act = tf.nn.relu, name='building_fc5')
        building_network = tl.layers.DenseLayer(building_network, n_units =   2, act = tf.nn.sigmoid, name='building_fc6')
        building_network = tl.layers.DenseLayer(building_network, n_units =   1, act = tf.nn.relu, name='building_fc7')
        self.building_predict_y = building_network.outputs
        self.building_cost = tl.cost.mean_squared_error(self.building_y, self.building_predict_y)
        self.building_optimize = tf.train.AdamOptimizer().minimize(self.building_cost, var_list=building_network.all_params)

        floor_x = self.locating_predict_y + self.building_predict_y
        floor_network = tl.layers.InputLayer(floor_x)
        floor_network = tl.layers.DenseLayer(floor_network, n_units = 128, act = tf.identity, name='floor_fc1')
        floor_network = tl.layers.DenseLayer(floor_network, n_units =  32, act = tf.nn.relu, name='floor_fc2')
        floor_network = tl.layers.DenseLayer(floor_network, n_units =  16, act = tf.nn.relu, name='floor_fc3')
        floor_network = tl.layers.DenseLayer(floor_network, n_units =   8, act = tf.nn.sigmoid, name='floor_fc4')
        floor_network = tl.layers.DenseLayer(floor_network, n_units =   4, act = tf.identity, name='floor_fc5')
        floor_network = tl.layers.DenseLayer(floor_network, n_units =   2, act = tf.identity, name='floor_fc6')
        floor_network = tl.layers.DenseLayer(floor_network, n_units =   1, act = tf.identity, name='floor_fc7')

        self.floor_predict_y = floor_network.outputs
        self.floor_cost = tl.cost.mean_squared_error(self.floor_y, self.floor_predict_y)
        self.floor_optimize = tf.train.AdamOptimizer().minimize(self.floor_cost, var_list=floor_network.all_params)

    def fit(self, x, y, epoch=2, batch_size=1024):
        # Data pre-processing
        self._preprocess(x, y)

        location_pair = np.concatenate((
            np.expand_dims(self.longitude_normalize_y, -1), np.expand_dims(self.latitude_normalize_y, -1)
        ), axis=-1)

        # Train the model
        print "<< training >>"
        self.sess.run(tf.global_variables_initializer())

        for k in range(5):
            print "-------- epoch ", k, ' ---------'
            print "\n< position >\n"
            for i in range(epoch):
                mini_x = data_helper.getMiniBatch(self.normalize_x , batch_size)
                mini_y = data_helper.getMiniBatch(location_pair, batch_size)
                feed_dict = {
                    self.x: mini_x.next(),
                    self.locating_y: mini_y.next()
                }
                _cost, _, _output = self.sess.run([self.locating_cost, self.locating_optimize, self.locating_predict_y], feed_dict=feed_dict)
                if i % 100 == 0:
                    print "epoch: ", i, '\tcost: ', _cost
                    # print 'y: ', _output
            # print 'y : ', feed_dict[self.locating_y][:5]
            # print 'y_: ', _output[:5]

            """
            print "\n< building >\n"

            for i in range(epoch):
                mini_x = data_helper.getMiniBatch(self.normalize_x, batch_size)
                mini_y = data_helper.getMiniBatch(np.expand_dims(self.buildingID_y, -1), batch_size)
                feed_dict = {
                    self.x: mini_x.next(),
                    self.building_y: mini_y.next()
                }
                _cost, _, _output = self.sess.run([self.building_cost, self.building_optimize, self.building_predict_y], feed_dict=feed_dict)
                if _cost < 0.00001:
                    break
                if i % 100 == 0:
                    print "epoch: ", i, '\tcost: ', _cost
            # print 'y : ', feed_dict[self.building_y][:5]
            # print 'y_: ', _output[:5]

            print "\n< floor >\n"
            
            for i in range(epoch):
                mini_x = data_helper.getMiniBatch(self.normalize_x, batch_size)
                mini_y = data_helper.getMiniBatch(np.expand_dims(self.floorID_y, -1), batch_size)
                feed_dict = {
                    self.x: mini_x.next(),
                    self.floor_y: mini_y.next()
                }
                _cost, _, _output = self.sess.run([self.floor_cost, self.floor_optimize, self.floor_predict_y], feed_dict=feed_dict)
                if _cost < 0.00001:
                    break
                if i % 100 == 0:
                    print "epoch: ", i, '\tcost: ', _cost
            # print 'y : ', feed_dict[self.floor_y][:5]
            # print 'y_: ', _output[:5]
            """
            
        self.save()

    def save(self, save_path='./complex_dnn.ckpt'):
        super(ComplexDNN, self).save()
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def predict(self, x, model_path='./complex_dnn.ckpt'):
        # Load model and parameter
        super(ComplexDNN, self).load()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        # Testing
        print "<< Testing >>"
        x = data_helper.normalizeX(x)
        predict_result = self.sess.run(self.locating_predict_y, feed_dict={self.x: x})
        predict_longitude = predict_result[:, 0]
        predict_latitude = predict_result[:, 1]
        predict_building = self.sess.run(self.building_predict_y, feed_dict={self.x: x})
        predict_floor = self.sess.run(self.floor_predict_y, feed_dict={self.x: x})

        # Reverse normalization
        predict_longitude, predict_latitude = data_helper.reverse_normalizeY(predict_longitude, predict_latitude)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1), 
            np.expand_dims(predict_latitude, axis=-1)), axis=-1)
        res = np.concatenate((res, predict_floor), axis=-1)
        res = np.concatenate((res, predict_building), axis=-1)
        return res

    def error(self, x, y, building_panality=50, floor_panality=4):
        _y = self.predict(x)
        building_error = len(y) - np.sum(np.equal(np.round(_y[:, 3]), y[:, 3]))
        floor_error = len(y) - np.sum(np.equal(np.round(_y[:, 2]), y[:, 2]))
        coordinates_error = np.sum(np.sqrt(
            np.square(_y[:, 0] - y[:, 0]), np.square(_y[:, 1] - y[:, 1])
        ))
        print building_error
        print floor_error
        print coordinates_error
        return building_panality * building_error + floor_panality * floor_error + coordinates_error