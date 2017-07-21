from abstract_model import AbstractModel
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import data_helper
import numpy as np

class EncoderDNN(AbstractModel):
    def __init__(self):
        self.input = Input((520,))
        self.encode_layer = Dense(256, activation='elu')(self.input)
        self.encode_layer = Dense(128, activation='elu')(self.encode_layer)
        decode_layer = Dense(256, activation='elu')(self.encode_layer)
        decode_layer = Dense(520, activation='elu')(decode_layer)
        self.encoder_model = Model(inputs=self.input, outputs=decode_layer)

        regression_net = Dense(256, activation='elu')(self.encode_layer)
        regression_net = Dense(128, activation='elu')(regression_net)
        regression_net = Dropout(0.5)(regression_net)
        regression_net = Dense(128, activation='elu')(regression_net)
        regression_net = Dropout(0.5)(regression_net)
        regression_net = Dense(64, activation='elu')(regression_net)
        regression_net = Dense(64, activation='elu')(regression_net)
        self.position_predict_output = Dense(2, activation='elu')(regression_net)
        self.regression_model = Model(inputs=self.input, outputs=self.position_predict_output)

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
        #self.longitude_normalize_y = np.expand_dims(self.longitude_normalize_y, -1)
        location_pair = np.concatenate((
            np.expand_dims(self.longitude_normalize_y, -1), np.expand_dims(self.latitude_normalize_y, -1)
        ), axis=-1)

        # Pre-train the encoder
        self.encoder_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.encoder_model.fit(self.normalize_x, self.normalize_x, epochs=50, batch_size=2024)
        print ""

        # Train Regression model
        self.regression_model.layers[2].trainable = False
        self.regression_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.regression_model.fit(self.normalize_x, location_pair, epochs=400, batch_size=512)
        
        # Train floor judgement model
        self.floor_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.floor_model.fit(self.normalize_x, data_helper.oneHotEncode(self.floorID_y), epochs=200, batch_size=512)
        
        # Train building ID judgement model
        self.building_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.building_model.fit(self.normalize_x, data_helper.oneHotEncode(self.buildingID_y), epochs=200, batch_size=512)
              
    def predict(self, x):
        x = data_helper.normalizeX(x)

        predict_result = self.regression_model.predict(x)
        predict_longitude = predict_result[:, 0]
        predict_latitude = predict_result[:, 1]
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
        print '\ncoor: ', coordinates_error
        print 'floor: ', floor_error
        print 'building: ', building_error
        print ''
        return building_panality * building_error + floor_panality * floor_error + coordinates_error