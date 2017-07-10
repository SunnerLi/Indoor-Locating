from abc import ABCMeta, abstractmethod
import numpy as np
import data_helper

class AbstractModel(object):
    __metaclass__ = ABCMeta

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

    def __init__(self):
        pass

    def _preprocess(self, x, y):
        self.normalize_x = data_helper.normalizeX(x)
        self.longitude_mean, self.longitude_std, self.longitude_normalize_y = \
            data_helper.normalizeY(y[:, 0])
        self.latitude_mean, self.latitude_std, self.latitude_normalize_y = \
            data_helper.normalizeY(y[:, 1])
        self.floor_y = y[:, 2]
        self.buildingID_y = y[:, 3]

    @abstractmethod
    def predict(self, x):
        pass

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