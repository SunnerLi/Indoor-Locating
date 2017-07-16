from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math

# Normalize scaler
longitude_scaler = MinMaxScaler()
latitude_scaler = MinMaxScaler()

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
    """
        Pre-processing toward the rssi feature.
        In my idea, I regard the value as 0 if original value is 100
        Second, the value will be transfer into positive one if original value is negative

        Arg:    arr - The training x array
        Ret:    The array after normalizing
    """
    res = np.copy(arr).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if res[i][j] == 100:
                res[i][j] = 0
            else:
                res[i][j] = -0.01 * res[i][j]
    return res

def normalizeY(longitude_arr, latitude_arr):
    """
        Use MinMaxScaler to normalize the longitude and latitude tag

        Arg:    longitude_arr   - The longitude array whose shape is [row_number]
                latitude_arr    - The latitude array whose shape is the same as the longitude array
        Ret:    The normalized longitude array and normalized latitude array
    """
    global longitude_scaler
    global latitude_scaler
    longitude_arr = np.reshape(longitude_arr, [-1, 1])
    latitude_arr = np.reshape(latitude_arr, [-1, 1])
    longitude_scaler.fit(longitude_arr)
    latitude_scaler.fit(latitude_arr)
    return np.reshape(longitude_scaler.transform(longitude_arr), [-1]), \
            np.reshape(latitude_scaler.transform(latitude_arr), [-1])

def reverse_normalizeY(longitude_arr, latitude_arr):
    """
        Recover the normalized longitude and latitude array to the origin

        Arg:    longitude_arr   - The normalized longitude array whose shape is [row_number]
                latitude_arr    - The normalized latitude array whose shape is the same as the longitude array
        Ret:    The original longitude array and original latitude array
    """
    global longitude_scaler
    global latitude_scaler
    longitude_arr = np.reshape(longitude_arr, [-1, 1])
    latitude_arr = np.reshape(latitude_arr, [-1, 1])
    return np.reshape(longitude_scaler.inverse_transform(longitude_arr), [-1]), \
            np.reshape(latitude_scaler.inverse_transform(latitude_arr), [-1])

def getMiniBatch(arr, batch_size=3):
    """
        Return the mini-batch array for the given batch size

        Arg:    arr         - The whole array that want to be sampled
                batch_size  - The batch size
        Ret:    The mini-batch array
    """
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