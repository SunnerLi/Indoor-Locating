from sklearn.externals import joblib
from sklearn.svm import SVC, SVR
import numpy as np
import data_helper

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
        joblib.dump(self.longitude_regression_model, self.longitude_regression_model_save_path)
        joblib.dump(self.latitude_regression_model, self.latitude_regression_model_save_path)
        joblib.dump(self.floor_classifier, self.floor_classifier_save_path)
        joblib.dump(self.building_classifier, self.building_classifier_save_path)

    def predict(self, x):
        # Load model
        print "<< Loading >>"
        self.longitude_regression_model = joblib.load(self.longitude_regression_model_save_path)
        self.latitude_regression_model = joblib.load(self.latitude_regression_model_save_path)
        self.floor_classifier = joblib.load(self.floor_classifier_save_path)
        self.building_classifier = joblib.load(self.building_classifier_save_path)

        # Testing
        print "<< Testing >>"
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
        res = np.concatenate((predict_longitude, predict_latitude))
        return predict_longitude, predict_latitude, predict_floor, predict_building
        
    def error(self, x, y):
        pass