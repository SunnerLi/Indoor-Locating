import numpy as np

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

x = np.reshape(range(40), [10, 4])
_gen = getMiniBatch(x)
for i in range(10):
    print 'list: ', _gen.next()