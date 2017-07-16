from ml_model import SVM, RandomForest, GradientBoostingDecisionTree
from dl_model import DNN
import numpy as np
import data_helper

train_csv_path = './TrainingData.csv'
valid_csv_path = './ValidationData.csv'

if __name__ == '__main__':
    # Load data
    train_x, train_y, valid_x, valid_y, test_x, test_y = \
        data_helper.load(train_csv_path, valid_csv_path)
    
    # Training
    svm_model = SVM()
    svm_model.fit(train_x, train_y)
    rf_model = RandomForest()
    rf_model.fit(train_x, train_y)
    gbdt_model = GradientBoostingDecisionTree()
    gbdt_model.fit(train_x, train_y)
    dnn_model = DNN()
    dnn_model.fit(train_x, train_y)

    # Print testing result
    print 'SVM eror: ', svm_model.error(test_x, test_y)
    print 'Random forest error: ', rf_model.error(test_x, test_y)
    print 'Gradient boosting decision tree error: ', gbdt_model.error(test_x, test_y)
    print 'DNN error: ', dnn_model.error(test_x, test_y)