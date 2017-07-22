from encoder_model import EncoderDNN
import numpy as np
import data_helper

train_csv_path = './TrainingData.csv'
valid_csv_path = './ValidationData.csv'

if __name__ == '__main__':
    # Load data
    train_x, train_y, valid_x, valid_y, test_x, test_y = \
        data_helper.load(train_csv_path, valid_csv_path)

    # Training
    encode_dnn_model = EncoderDNN()    
    encode_dnn_model.fit(train_x, train_y)
    print(test_y)
    print(encode_dnn_model.predict(test_x))
    print('DNN error: ', encode_dnn_model.error(test_x, test_y))
    