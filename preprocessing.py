from pywt import wavedec
import numpy as np
import multiprocessing as mp

from math import fabs
from random import shuffle

from common import qrs_detect

__MAPPING__ = {
    'A': 0,
    'N': 1,
    'O': 2,
    '~': 3
}

__REVERSE_MAPPING__ = {
    0: 'A',
    1: 'N',
    2: 'O',
    3: '~'
}


def normalize(X):
    pool = mp.Pool()
    x_new = pool.map(normalize_ecg, X)
    pool.close()
    return x_new


def normalize_ecg(ecg):
    ecg = qrs_detect.remove_dc_component(ecg)
    return qrs_detect.normalize_ecg(ecg)


def format_labels(labels):
    return [__MAPPING__[x] for x in labels]


def get_original_label(category):
    return __REVERSE_MAPPING__[category]


def convolution(signal, window_size, step, fadein=0, fadeout=0):
    """
    Splits the signal into a list of signals of size=window_size
    by applying moving window(convolution) on the signal with step=step
    :param signal: input signal
    :param window_size: convolution window
    :param step: convolution step
    :param fadein: number of points to be ignored at the beginning of the signal
    :param fadeout: number of points to be ignored at the end of the signal
    :return: list of generated signals
    """
    start = fadein
    output = []
    end = len(signal) - fadeout
    while start + window_size < end:
        output.append(signal[start:start + window_size])
        start += step

    return output


def denoise(row, lvl):
    info = wavedec(row, 'db1', level=lvl)[0]
    return info


def shuffle_data(data, labels):
    """
    Shuffles input data

    In some cases input data might be distributed sorted which might create a hidden error
    in training/validation process so it's better to always shuffle input data before usage
    :return: Shuffled input data
    """
    data_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(data)))
    shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        labels_shuf.append(labels[i])
    return (np.array(data_shuf), labels_shuf)


def balance_data(X, Y, class_weights):
    nX = []
    nY = []

    examples = dict()
    for key in class_weights.keys():
        examples[key] = [x for (x, y) in zip(X, Y) if y == key]

    for key in examples.keys():
        weight = class_weights[key]
        while weight >= 1.0:
            nX += examples[key]
            nY += [key for i in range(len(examples[key]))]
            weight -= 1
        extra = int(weight * len(examples[key]))
        if extra > 0:
            nX += examples[key][:extra]
            nY += [key for i in range(extra)]

    return shuffle_data(nX, nY)


def embedding(row):
    """
    Normalizes the data base on max value into array in range of [0. 1.]
    :param row: original data
    :return: array of length of row normalized in range of [0. 1.]
    """
    amax = np.amax(row)
    amin = np.amin(row)
    abs = float(max(fabs(amin), fabs(amax)))
    return np.array([x / abs for x in row])
