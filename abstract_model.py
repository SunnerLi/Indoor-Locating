from abc import ABCMeta, abstractmethod
from sklearn.externals import joblib
import numpy as np
import data_helper
import pickle

class AbstractModel(object):
    __metaclass__ = ABCMeta

    # Model save path
    parameter_save_path = 'param.pkl'
    longitude_regression_model_save_path = None
    latitude_regression_model_save_path = None
    floor_classifier_save_path = None
    building_classifier_save_path = None

    # ML model object
    longitude_regression_model = None
    latitude_regression_model = None
    floor_classifier = None
    building_classifier = None

    # Normalize variable
    longitude_mean = None
    longitude_std = None
    latitude_mean = None
    latitude_std = None
    longitude_shift_distance = None
    latitude_shift_distance = None

    # Training data
    normalize_x = None
    longitude_normalize_y = None
    latitude_normalize_y = None
    floor_y = None
    buildingID_y = None

    def __init__(self):
        pass

    def _preprocess(self, x, y):
        """
            Data pre-processing for the x and y array
            It's not recommend to use this function directly!

            Arg:    x   - The feature array
                    y   - The tag array
        """
        self.normalize_x = data_helper.normalizeX(x)
        self.longitude_normalize_y, self.latitude_normalize_y = data_helper.normalizeY(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

    def save(self):
        """
            Save the training result
        """
        print "<< Saving >>"
        joblib.dump(self.longitude_regression_model, self.longitude_regression_model_save_path)
        joblib.dump(self.latitude_regression_model, self.latitude_regression_model_save_path)
        joblib.dump(self.floor_classifier, self.floor_classifier_save_path)
        joblib.dump(self.building_classifier, self.building_classifier_save_path)

    def load(self):
        """
            Load the pre-trained model
        """
        self.longitude_regression_model = joblib.load(self.longitude_regression_model_save_path)
        self.latitude_regression_model = joblib.load(self.latitude_regression_model_save_path)
        self.floor_classifier = joblib.load(self.floor_classifier_save_path)
        self.building_classifier = joblib.load(self.building_classifier_save_path)

    def fit(self, x, y):
        """
            Train the model

            Arg:    x   - The feature array
                    y   - The tag array
        """
        # Data pre-processing
        self._preprocess(x, y)

        # Train the model
        print "<< training >>"
        self.longitude_regression_model.fit(self.normalize_x, self.longitude_normalize_y)
        self.latitude_regression_model.fit(self.normalize_x, self.latitude_normalize_y)
        self.floor_classifier.fit(self.normalize_x, self.floorID_y)
        self.building_classifier.fit(self.normalize_x, self.buildingID_y)

        # Release the memory
        del self.normalize_x
        del self.longitude_normalize_y
        del self.latitude_normalize_y
        del self.floorID_y
        del self.buildingID_y

        # Save the result
        self.save()

    def predict(self, x):
        """
            Predict the tag by the model
            You should train the model before calling this function

            Arg:    x   - The feature array that you want to predict
            Ret:    The predict result whose shape is [num_of_row, 4]
        """
        # Load model
        self.load()

        # Testing
        x = data_helper.normalizeX(x)
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)
        predict_floor = self.floor_classifier.predict(x)
        predict_building = self.building_classifier.predict(x)

        # Reverse normalization
        predict_longitude, predict_latitude = data_helper.reverse_normalizeY(predict_longitude, predict_latitude)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1), 
            np.expand_dims(predict_latitude, axis=-1)), axis=-1)
        res = np.concatenate((res, np.expand_dims(predict_floor, axis=-1)), axis=-1)
        res = np.concatenate((res, np.expand_dims(predict_building, axis=-1)), axis=-1)
        return res

    def error(self, x, y, building_panality=50, floor_panality=4):
        """
            Return the error by the predict result
            The formula of error computing is referred from the Kaggle information

            < formula >
            Error = building_penality * building_error + floor_penality * floor_error + coordinates_error

            Arg:    x                   - The feature array that you want to test
                    y                   - The ground truth array
                    building_panality   - The coefficient that is defined in the error formula
                    floor_panality      - The coefficient that is defined in the error formula
            Ret:    The value of error
        """
        _y = self.predict(x)
        building_error = len(y) - np.sum(np.equal(np.round(_y[:, 3]), y[:, 3]))
        floor_error = len(y) - np.sum(np.equal(np.round(_y[:, 2]), y[:, 2]))
        longitude_error = np.sum(np.sqrt(np.square(_y[:, 0] - y[:, 0])))
        latitude_error = np.sum(np.sqrt(np.square(_y[:, 1] - y[:, 1])))
        coordinates_error = longitude_error + latitude_error
        return building_panality * building_error + floor_panality * floor_error + coordinates_error