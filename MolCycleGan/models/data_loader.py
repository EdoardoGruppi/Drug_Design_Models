import copy

import numpy as np
import pandas as pd


def load_data(path, input_shape):
    values = pd.read_csv(path, index_col=0).values
    values = values.reshape((-1,) + input_shape)
    return values


def minibatch(data, batch_size):
    data = copy.copy(data)
    length = data.shape[0]
    np.random.shuffle(data)
    epoch = 0
    i = 0

    while True:
        if i + batch_size > length:
            np.random.shuffle(data)
            i = 0
            epoch += 1
        images_batch = data[i:(i + batch_size), :]
        i += batch_size
        yield epoch, images_batch


def minibatchAB(dataA, dataB, batch_size):
    print('Each epoch is made up of {} iters'.format(len(dataA)//batch_size))
    batchA = minibatch(dataA, batch_size)
    batchB = minibatch(dataB, batch_size)

    while True:
        ep1, A = next(batchA)
        ep2, B = next(batchB)
        yield min(ep1, ep2), A, B
