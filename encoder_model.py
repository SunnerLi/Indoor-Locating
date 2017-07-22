from abstract_model import AbstractModel
from keras.layers import Dense, Input, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model
import data_helper
import numpy as np

class EncoderDNN(AbstractModel):
    def __init__(self):
        self.input = Input((520,))
        
        self.encode_layer = Dense(256, activation='elu')(self.input)
        self.encode_layer = Dense(64, activation='elu')(self.encode_layer)
        decode_layer = Dense(256, activation='elu')(self.encode_layer)
        decode_layer = Dense(520, activation='elu')(decode_layer)
        self.encoder_model = Model(inputs=self.input, outputs=decode_layer)

        longitude_regression_net = Dense(256, activation='elu')(self.encode_layer)
        longitude_regression_net = Dense(128, activation='elu')(longitude_regression_net)
        longitude_regression_net = Dropout(0.5)(longitude_regression_net)
        longitude_regression_net = Dense(128, activation='elu')(longitude_regression_net)
        longitude_regression_net = Dropout(0.5)(longitude_regression_net)
        longitude_regression_net = Dense(64, activation='elu')(longitude_regression_net)
        longitude_regression_net = Dense(64, activation='elu')(longitude_regression_net)
        self.longitude_predict_output = Dense(1, activation='elu')(longitude_regression_net)
        self.longitude_regression_model = Model(inputs=self.input, outputs=self.longitude_predict_output)

        latitude_regression_net = Dense(256, activation='elu')(self.encode_layer)
        latitude_regression_net = Dense(128, activation='elu')(latitude_regression_net)
        latitude_regression_net = Dropout(0.5)(latitude_regression_net)
        latitude_regression_net = Dense(128, activation='elu')(latitude_regression_net)
        latitude_regression_net = Dropout(0.5)(latitude_regression_net)
        latitude_regression_net = Dense(64, activation='elu')(latitude_regression_net)
        latitude_regression_net = Dense(64, activation='elu')(latitude_regression_net)
        self.latitude_predict_output = Dense(1, activation='elu')(latitude_regression_net)
        self.latitude_regression_model = Model(inputs=self.input, outputs=self.latitude_predict_output)

        # merge_layer = Concatenate([self.longitude_predict_output, self.latitude_predict_output])
        floor_net = Dense(256, activation='elu')(self.encode_layer)
        floor_net = Dense(128, activation='elu')(floor_net)
        self.floor_predict_output = Dense(5, activation='elu')(floor_net)
        self.floor_model = Model(inputs=self.input, outputs=self.floor_predict_output)

        building_net = Dense(64, activation='elu')(self.encode_layer)
        building_net = Dense(32, activation='elu')(building_net)
        self.buildingID_predict_output = Dense(3, activation='elu')(building_net)
        self.building_model = Model(inputs=self.input, outputs=self.buildingID_predict_output)

    def fit(self, x, y):
        # Data pre-processing
        self._preprocess(x, y)
        self.longitude_normalize_y = np.expand_dims(self.longitude_normalize_y, -1)
        self.latitude_normalize_y = np.expand_dims(self.latitude_normalize_y, -1)

        # Pre-train the encoder
        self.encoder_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.encoder_model.fit(self.normalize_x, self.normalize_x, epochs=50, batch_size=512)
        print("")

        # Disable encoder layer trainable properties
        for i in range(len(self.encoder_model.layers)):
            self.longitude_regression_model.layers[i].trainable = False
            self.latitude_regression_model.layers[i].trainable = False

        # Train longitude regression model
        self.longitude_regression_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.latitude_regression_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.longitude_regression_model.fit(self.normalize_x, self.longitude_normalize_y, epochs=100, batch_size=512)

        # Train latitude regression model
        for i in range(len(self.longitude_regression_model.layers)):
            self.longitude_regression_model.layers[i].trainable = False
        self.latitude_regression_model.fit(self.normalize_x, self.latitude_normalize_y, epochs=100, batch_size=512)
        
        # Train floor judgement model 
        self.floor_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.floor_model.fit(self.normalize_x, data_helper.oneHotEncode(self.floorID_y), epochs=500, batch_size=512)
        
        # Train building ID judgement model
        self.building_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.building_model.fit(self.normalize_x, data_helper.oneHotEncode(self.buildingID_y), epochs=200, batch_size=512)
              
    def predict(self, x):
        x = data_helper.normalizeX(x)
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)
        predict_floorID = self.floor_model.predict(x)
        predict_buildingID = self.building_model.predict(x)
        
        # Reverse normalization
        predict_longitude, predict_latitude = data_helper.reverse_normalizeY(predict_longitude, predict_latitude)
        predict_floorID = data_helper.oneHotDecode(predict_floorID)
        predict_buildingID = data_helper.oneHotDecode(predict_buildingID)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1), 
            np.expand_dims(predict_latitude, axis=-1)), axis=-1)
        res = np.concatenate((res, np.expand_dims(predict_floorID, axis=-1)), axis=-1)
        res = np.concatenate((res, np.expand_dims(predict_buildingID, axis=-1)), axis=-1)
        return res

    def error(self, x, y, building_panality=50, floor_panality=4):
        _y = self.predict(x)
        building_error = len(y) - np.sum(np.equal(np.round(_y[:, 3]), y[:, 3]))
        floor_error = len(y) - np.sum(np.equal(np.round(_y[:, 2]), y[:, 2]))
        longitude_error = np.sum(np.sqrt(np.square(_y[:, 0] - y[:, 0])))
        latitude_error = np.sum(np.sqrt(np.square(_y[:, 1] - y[:, 1])))
        coordinates_error = longitude_error + latitude_error

        print('long: ', longitude_error)
        print('lat : ', latitude_error)
        print('\ncoor: ', coordinates_error)
        print('floor: ', floor_error)
        print('building: ', building_error)
        print('')
        return building_panality * building_error + floor_panality * floor_error + coordinates_error