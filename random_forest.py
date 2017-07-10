from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abstract_model import AbstractModel
from sklearn.externals import joblib
import numpy as np
import data_helper
import pickle

class RandomForest(AbstractModel):
    # ML model object
    longitude_regression_model = None
    latitude_regression_model = None
    floor_classifier = None
    building_classifier = None

    # Model save path
    parameter_save_path = 'rf_param.pkl'
    longitude_regression_model_save_path = './rf_long.pkl'
    latitude_regression_model_save_path = './rf_lat.pkl'
    floor_classifier_save_path = './rf_floor.pkl'
    building_classifier_save_path = './rf_building.pkl'

    def __init__(self):
        self.longitude_regression_model = RandomForestRegressor()
        self.latitude_regression_model = RandomForestRegressor()
        self.floor_classifier = RandomForestClassifier()
        self.building_classifier = RandomForestClassifier()

    def fit(self, x, y):
        # Data pre-processing
        self._preprocess(x, y)

        # Train the model
        print "<< training >>"
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