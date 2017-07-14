import numpy as np

def oneHotEncode(arr):
    import pandas as pd
    return pd.get_dummies(arr).values

def oneHotDecode(arr):
    return np.argmax(np.round(arr), axis=1)

a = np.asarray([2, 1, 2, 0])
a = oneHotEncode(a)
print oneHotDecode(a)