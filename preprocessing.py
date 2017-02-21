from pywt import wavedec
import numpy as np


def format_labels(labels):
    original_labels = list(set(labels))
    original_labels.sort()

    labels = [original_labels.index(x) for x in labels]
    mapping = dict()
    for i in range(len(original_labels)):
        mapping[i] = original_labels[i]
    return (labels, mapping)


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
    end = signal.shape[1] - fadeout
    while start + window_size < end:
        output.append(signal[0:1, start:start + window_size])
        start += step

    return output


def denoise(row, lvl):
    info = wavedec(row, 'db1', level=lvl)[0]
    return np.reshape(info, (-1, 1))
