import numpy as np


def find(a, condition):
    """
    Analog to matlab: find(array condition)
    eg find(array == 0) -> find(array, lambda x: x == 0)
    :return: array of positions where condition is true
    """
    return [i for (i, val) in enumerate(a) if condition(val)]


def diff(a):
    """
    Analog to matlab: diff(array)
    """
    return a[1:] - a[:-1]


def add(array, value):
    """
    Analog to matlab: array + value
    """
    return np.array([x + value for x in array])


def np_max(array):
    """
    Analog to matlab: max(array)
    :return: tuple of (max value, position)
    """
    idx = np.argmax(array)
    return (array[idx], idx)


def np_min(array):
    """
    Analog to matlab: min(array)
    :return: tuple of (min value, position)
    """
    idx = np.argmin(array)
    return (array[idx], idx)
