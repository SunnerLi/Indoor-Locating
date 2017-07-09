from sklearn.svm import SVC, SVR
import numpy as np

class SVM(object):
    longitude_regression_model = None
    latitude_regression_model = None
    floor_classifier = None
    building_classifier = None

    def __init__(self):
        self.longitude_regression_model = SVR(verbose=True)
        self.latitude_regression_model = SVR(verbose=True)
        self.floor_classifier = SVC(verbose=True)
        self.building_classifier = SVC(verbose=True)

    def fit(self, x, y):
        print np.shape(y[:, 0])
        print "<< train longitude regression model >>"
        self.longitude_regression_model.fit(x, y[:, 0])

        """
        print "<< train latitude regression model >>"
        self.latitude_regression_model.fit(x, y[:, 1])
        print "<< train floor classifier >>"
        self.longitude_regression_model.fit(x, y[:, 2])
        print "<< train building classifier >>"
        self.longitude_regression_model.fit(x, y[:, 3])
        """
