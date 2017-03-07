import numpy as np
from math import *

from scipy.signal import lfilter


def qrs_detect(ecg):
    """
    Based on this article
    http://cnx.org/contents/YR1BUs9_@1/QRS-Detection-Using-Pan-Tompki

    :param ecg: ECG signal
    :param fs: signal frequency
    :return: tuple of 3 arrays (positions of Q, positions of R, positions of S)
    """
    ecg1 = cancel_dc_drift(ecg)
    ecg2 = low_pass_filtering(ecg1)
    ecg3 = high_pass_filtering(ecg2)
    ecg4 = derivative_filter(ecg3)
    ecg5 = squaring(ecg4)
    ecg6 = moving_window_integration(ecg5)
    q, r, s = qrs(ecg1, ecg6)
    return (q, r, s)


def cancel_dc_drift(ecg):
    mean = np.mean(ecg)
    # cancel DC components
    ecg = [x - mean for x in ecg]
    # normalize to 1
    abs_max = max([fabs(x) for x in ecg])
    return np.array([x / abs_max for x in ecg])


def low_pass_filtering(ecg):
    # LPF (1-z^-6)^2/(1-z^-1)^2
    b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
    a = [1, -2, 1]

    # transfter function of LPF
    h_LP = lfilter(b, a, np.append([1], np.zeros(12)))

    ecg2 = np.convolve(ecg, h_LP)
    # cancel delay
    # ecg2 = ecg2 (range(6, len(ecg) + 6))
    abs_max = max([fabs(x) for x in ecg2])
    return np.array([x / abs_max for x in ecg2])


def high_pass_filtering(ecg):
    # HPF = Allpass-(Lowpass) = z^-16-[(1-z^32)/(1-z^-1)]
    b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 32, -32, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    a = [1, -1]

    # impulse response iof HPF
    h_HP = lfilter(b, a, np.append([1], np.zeros(32)))
    ecg3 = np.convolve(ecg, h_HP)
    # cancel delay
    # ecg3 = ecg3 (range(16, len(ecg) + 16))
    abs_max = max([fabs(x) for x in ecg3])
    return np.array([x / abs_max for x in ecg3])


def derivative_filter(ecg):
    # Make impulse response
    h = [-1, -2, 0, 2, 1]
    h = [x / 8 for x in h]
    # Apply filter
    ecg4 = np.convolve(ecg, h)
    # ecg4 = ecg4(range(2, len(ecg) + 2))
    ecg4 = np.roll(ecg4, 2)
    abs_max = max([fabs(x) for x in ecg4])
    return np.array([x / abs_max for x in ecg4])


def squaring(ecg):
    ecg5 = np.square(ecg)
    abs_max = max([fabs(x) for x in ecg5])
    return np.array([x / abs_max for x in ecg5])


def moving_window_integration(ecg):
    # Make impulse response
    h = np.ones(31)
    h = np.array([x / 31 for x in h])

    # Apply filter
    ecg6 = np.convolve(ecg, h)
    # ecg6 = ecg6(range(15, len(ecg) + 15))
    ecg6 = np.roll(ecg6, 15)
    abs_max = max([fabs(x) for x in ecg6])
    return np.array([x / abs_max for x in ecg6])


def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def diff(a):
    return a[1:] - a[:-1]


def add(array, value):
    return np.array([x + value for x in array])


def np_max(array):
    idx = np.argmax(array)
    return (array[idx], idx)


def np_min(array):
    idx = np.argmin(array)
    return (array[idx], idx)


def qrs(ecg1, ecg6):
    max_h = max(ecg6)
    thresh = np.mean(ecg6)
    poss_reg = np.transpose(np.select([ecg6 > thresh * max_h], [ecg6]))

    left = find(diff(np.append([0], poss_reg)), lambda x: x == 1)
    right = find(diff(np.append(poss_reg, [0])), lambda x: x == 1)

    # cancel delay because of LP and HP
    shift = -(6 + 16)
    left = add(left, shift)
    # cancel delay because of LP and HP
    right = add(right, shift)

    R_values = []
    R_locs = []
    Q_values = []
    Q_locs = []
    S_values = []
    S_locs = []
    for i in range(len(left)):
        R_value, R_loc = np_max(ecg1[left[i]:right[i]])
        # add offset
        R_loc = R_loc - 1 + left[i]
        R_values.append(R_value)
        R_locs.append(R_loc)

        Q_value, Q_loc = np_min(ecg1[left[i]:R_loc])
        Q_loc = Q_loc - 1 + left[i]
        Q_values.append(Q_value)
        Q_locs.append(Q_loc)

        S_value, S_loc = np_min(ecg1[left[i]:right[i]])
        S_loc = S_loc - 1 + left[i]
        S_values.append(S_value)
        S_locs.append(S_loc)

    Q_locs = [Q_locs[i] for i in find(Q_locs, lambda x: x != 0)]
    R_locs = [R_locs[i] for i in find(R_locs, lambda x: x != 0)]
    S_locs = [S_locs[i] for i in find(S_locs, lambda x: x != 0)]
    return (Q_locs, R_locs, S_locs)
