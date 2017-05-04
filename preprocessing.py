from collections import Counter

from pywt import wavedec
import numpy as np
import multiprocessing as mp

from math import fabs
from random import shuffle

from common import qrs_detect
from common.qrs_detect import pqrst_detect
from utils import async
from utils import matlab

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


def numpy_set_length(a, length, value=0):
    size = a.shape[0]
    if size > length:
        return a[:length]
    elif size == length:
        return a
    else:
        diff = length - size
        append = np.zeros(diff, dtype=np.float32)
        append.fill(value)
        return np.append(a, append)


def transpose_if_needed(row):
    """
    :param row: 
    :return: (row data, whether the data was transposed)
    """
    p, q, r, s, t = pqrst_detect(row)

    qs = [abs(row[x] - row[y]) for x, y in zip(q, s)]
    m = [max(abs(row[p1]), abs(row[p2]), abs(row[p3])) for p1, p2, p3 in zip(q, r, s)]
    rotate_points = sum([1 if x / y > 0.7 else 0 for x, y in zip(qs, m)])
    if rotate_points > int(0.6 * len(qs)) + 1:
        return transpose_ecg(row), True
    return row, False


def transpose_ecg(ecg):
    return np.array([-1 * x for x in ecg])


def trimboth(row, portion):
    filter = portion * max(
        fabs(np.amin(row)),
        abs(np.amax(row))
    )

    return np.array([x if -filter < x < filter else 0 for x in row])


def normalize(X):
    pool = mp.Pool()
    x_new = pool.map(normalize_ecg, X)
    pool.close()
    return x_new


def normalize_ecg(ecg):
    ecg = qrs_detect.remove_dc_component(ecg)
    ecg = qrs_detect.normalize_ecg(ecg)
    # ecg = transpose_if_needed(ecg)
    return ecg


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


def show_balancing(y):
    counter = Counter(y)
    for key in sorted(list(counter.keys())):
        print(key, counter[key])


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


def balance(x, y):
    uniq = np.unique(y)

    selected = dict()

    for val in uniq:
        selected[val] = [x[i] for i in matlab.find(y, lambda v: v == val)]

    min_len = min([len(x) for x in selected.values()])

    x = []
    y = []
    for (key, value) in selected.items():
        x += value[:min_len]
        y += [key for i in range(min_len)]

    x, y = shuffle_data(x, y)

    return x, y


def balance2(x, y):
    uniq = np.unique(y)

    selected = dict()

    for val in uniq:
        selected[val] = [x[i] for i in matlab.find(y, lambda v: v == val)]

    min_len = 6 * min([len(x) for x in selected.values()])

    x = []
    y = []
    for (key, value) in selected.items():
        slen = min(len(value), min_len)
        x += value[:slen]
        y += [key for i in range(slen)]

    x, y = shuffle_data(x, y)

    return x, y
