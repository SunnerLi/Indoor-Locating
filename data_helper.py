import pandas as pd
import numpy as np
import math

def load(train_file_name, valid_file_name):
    """
        Load the training data and validation data
        In this function, 1/10 part of training will be the 'final' validation data.
        The other 9/10 part of training will be the 'final' training data.
        At last, the validation data will be the 'final' testing data.

        The x data contain 520 rssi value.
        The y data contain the Longitude, Latitude, floor ID and building ID.

        Arg:    train_file_name - The name of the training file
                valid_file_name - The name of the validation file
        Ret:    training x, training y, validation x, validation y, testing x and testing y data
    """
    # Read the file
    if train_file_name == None or valid_file_name == None:
        print 'file name is None...'
        exit()
    train_data_frame = pd.read_csv(train_file_name)
    test_data_frame = pd.read_csv(valid_file_name)

    # Random pick 1/10 data to be the final validation data
    rest_data_frame = train_data_frame
    valid_data_trame = pd.DataFrame(columns=train_data_frame.columns)
    valid_num = int(len(train_data_frame)/10)
    sample_row = rest_data_frame.sample(valid_num)
    rest_data_frame = rest_data_frame.drop(sample_row.index)
    valid_data_trame = valid_data_trame.append(sample_row)
    train_data_frame = rest_data_frame

    # Split data frame and return
    training_x = train_data_frame.get_values().T[:520].T
    training_y = train_data_frame.get_values().T[[520, 521, 522, 523], :].T
    validation_x = valid_data_trame.get_values().T[:520].T
    validation_y = valid_data_trame.get_values().T[[520, 521, 522, 523], :].T
    testing_x = test_data_frame.get_values().T[:520].T
    testing_y = test_data_frame.get_values().T[[520, 521, 522, 523], :].T
    return training_x, training_y, validation_x, validation_y, testing_x, testing_y

def normalizeX(arr):
    res = np.copy(arr).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if res[i][j] == 100:
                res[i][j] = 0
            else:
                res[i][j] = -0.01 * res[i][j]
    return res

def normalizeY(arr):
    _mean = np.mean(arr)
    _std = np.std(arr)
    shift_distance = math.floor(np.min((arr - _mean) / _std))
    return _mean, _std, shift_distance, (arr - _mean) / _std - shift_distance

def reverse_normalizeY(arr, mean, std, shift_distance):
    return (arr + shift_distance) * std + mean

def getMiniBatch(arr, batch_size=3):
    index = 0
    while True: 
        # print index + batch_size
        if index + batch_size >= len(arr):
            res = arr[index:]
            res = np.concatenate((res, arr[:index + batch_size - len(arr)]))
        else:
            res = arr[index:index + batch_size]
        index = (index + batch_size) % len(arr)
        yield res