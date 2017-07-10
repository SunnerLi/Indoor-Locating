from sklearn.externals import joblib
from sklearn.svm import SVC, SVR
import numpy as np
import data_helper
import pickle

class SVM(object):
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

    # Training data
    normalize_x = None
    longitude_normalize_y = None
    latitude_normalize_y = None
    floor_y = None
    buildingID_y = None

    # Model save path
    parameter_save_path = 'param.pkl'
    longitude_regression_model_save_path = './long.pkl'
    latitude_regression_model_save_path = './lat.pkl'
    floor_classifier_save_path = './floor.pkl'
    building_classifier_save_path = './building.pkl'

    def __init__(self):
        self.longitude_regression_model = SVR(verbose=True)
        self.latitude_regression_model = SVR(verbose=True)
        self.floor_classifier = SVC(verbose=True)
        self.building_classifier = SVC(verbose=True)

    def __preprocess(self, x, y):
        self.normalize_x = data_helper.normalizeX(x)
        self.longitude_mean, self.longitude_std, self.longitude_normalize_y = \
            data_helper.normalizeY(y[:, 0])
        self.latitude_mean, self.latitude_std, self.latitude_normalize_y = \
            data_helper.normalizeY(y[:, 1])
        self.floor_y = y[:, 2]
        self.buildingID_y = y[:, 3]

    def fit(self, x, y):
        # Data pre-processing
        self.__preprocess(x, y)

        # Train the model
        print "<< training >>"
        #print np.shape(self.normalize_x), np.shape(self.longitude_normalize_y)
        self.longitude_regression_model.fit(self.normalize_x, self.longitude_normalize_y)
        self.latitude_regression_model.fit(self.normalize_x, self.latitude_normalize_y)
        self.floor_classifier.fit(self.normalize_x, self.floor_y)
        self.building_classifier.fit(self.normalize_x, self.buildingID_y)

        # Release the memory
        del self.normalize_x
        del self.longitude_normalize_y
        del self.latitude_normalize_y
        del self.floor_y
        del self.buildingID_y

        # Save the result
        print "<< Saving >>"
        with open(self.parameter_save_path, 'wb') as f:
            para_dict = {
                'longitude_mean': self.longitude_mean,
                'longitude_std': self.longitude_std,
                'latitude_mean': self.latitude_mean,
                'latitude_std': self.latitude_std
            }
            pickle.dump(para_dict, f)
        joblib.dump(self.longitude_regression_model, self.longitude_regression_model_save_path)
        joblib.dump(self.latitude_regression_model, self.latitude_regression_model_save_path)
        joblib.dump(self.floor_classifier, self.floor_classifier_save_path)
        joblib.dump(self.building_classifier, self.building_classifier_save_path)

    def predict(self, x):
        # Load model
        print "<< Loading >>"
        with open(self.parameter_save_path, 'rb') as f:
            para_dict = pickle.load(f)
            self.longitude_mean = para_dict['longitude_mean']
            self.longitude_std = para_dict['longitude_std']
            self.latitude_mean = para_dict['latitude_mean']
            self.latitude_std = para_dict['latitude_std']
        self.longitude_regression_model = joblib.load(self.longitude_regression_model_save_path)
        self.latitude_regression_model = joblib.load(self.latitude_regression_model_save_path)
        self.floor_classifier = joblib.load(self.floor_classifier_save_path)
        self.building_classifier = joblib.load(self.building_classifier_save_path)

        # Testing
        print "<< Testing >>"
        x = data_helper.normalizeX(x)
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)
        predict_floor = self.floor_classifier.predict(x)
        predict_building = self.building_classifier.predict(x)

        # Reverse normalization
        predict_longitude = data_helper.reverse_normalizeY(
            predict_longitude, self.longitude_mean, self.longitude_std
        )
        predict_latitude = data_helper.reverse_normalizeY(
            predict_latitude, self.latitude_mean, self.latitude_std
        )

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1), 
            np.expand_dims(predict_latitude, axis=-1)), axis=-1)
        res = np.concatenate((res, np.expand_dims(predict_floor, axis=-1)), axis=-1)
        res = np.concatenate((res, np.expand_dims(predict_building, axis=-1)), axis=-1)
        return res
        
    def error(self, x, y, building_panality=50, floor_panality=4):
        _y = self.predict(x)
        building_error = np.sum(np.equal(_y[3], y[3]))
        floor_error = np.sum(np.equal(_y[2], y[2]))
        coordinates_error = np.sum(np.sqrt(
            np.square(_y[0] - y[0]), np.square(_y[1] - y[1])
        ))
        print building_error
        print floor_error
        print coordinates_error
        return building_panality * building_error + floor_panality * floor_error + coordinates_error