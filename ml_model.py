from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abstract_model import AbstractModel
from sklearn.externals import joblib
from sklearn.svm import SVC, SVR
import numpy as np
import data_helper
import pickle

"""
    This program define 3 wrapper toward the fundemential model.
    They are SVM, random forest and gradient boosting decision tree.
    They inherit the properties and methods of AbstractModel
"""

class SVM(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './svm_long.pkl'
    latitude_regression_model_save_path = './svm_lat.pkl'
    floor_classifier_save_path = './svm_floor.pkl'
    building_classifier_save_path = './svm_building.pkl'

    def __init__(self):
        self.longitude_regression_model = SVR(verbose=True)
        self.latitude_regression_model = SVR(verbose=True)
        self.floor_classifier = SVC(verbose=True)
        self.building_classifier = SVC(verbose=True)

class RandomForest(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './rf_long.pkl'
    latitude_regression_model_save_path = './rf_lat.pkl'
    floor_classifier_save_path = './rf_floor.pkl'
    building_classifier_save_path = './rf_building.pkl'

    def __init__(self):
        self.longitude_regression_model = RandomForestRegressor()
        self.latitude_regression_model = RandomForestRegressor()
        self.floor_classifier = RandomForestClassifier()
        self.building_classifier = RandomForestClassifier()

class GradientBoostingDecisionTree(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './gb_long.pkl'
    latitude_regression_model_save_path = './gb_lat.pkl'
    floor_classifier_save_path = './gb_floor.pkl'
    building_classifier_save_path = './gb_building.pkl'

    def __init__(self):
        self.longitude_regression_model = GradientBoostingRegressor()
        self.latitude_regression_model = GradientBoostingRegressor()
        self.floor_classifier = GradientBoostingClassifier()
        self.building_classifier = GradientBoostingClassifier()