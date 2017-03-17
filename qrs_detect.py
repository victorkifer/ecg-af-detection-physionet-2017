from math import *

from utils.matlab import *

np.set_printoptions(threshold=np.nan)

from scipy.signal import lfilter


def remove_dc_component(ecg):
    mean = np.mean(ecg)
    # cancel DC components
    return np.array([x - mean for x in ecg])


def normalize_ecg(ecg):
    """
    Normalizes to a range of [-1; 1]
    :param ecg: input signal
    :return: normalized signal
    """
    abs_max = max([fabs(x) for x in ecg])
    return np.array([x / abs_max for x in ecg])


def r_detect(ecg):
    """
    Based on this article
    http://cnx.org/contents/YR1BUs9_@1/QRS-Detection-Using-Pan-Tompki

    :param ecg: ECG signal
    :param fs: signal frequency
    :return: list, positions of R
    """
    ecg2 = low_pass_filtering(ecg)
    ecg3 = high_pass_filtering(ecg2)
    ecg4 = derivative_filter(ecg3)
    ecg5 = squaring(ecg4)
    ecg6 = moving_window_integration(ecg5)
    left, right = left_right(ecg6)
    return qrs(ecg, left, right)


def low_pass_filtering(ecg):
    # LPF (1-z^-6)^2/(1-z^-1)^2
    b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
    a = [1, -2, 1]

    # transfer function of LPF
    h_LP = lfilter(b, a, np.append([1], np.zeros(12)))

    ecg2 = np.convolve(ecg, h_LP)
    # cancel delay
    ecg2 = np.roll(ecg2, -6)
    return normalize_ecg(ecg2)


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
    ecg3 = np.roll(ecg3, -16)
    return normalize_ecg(ecg3)


def derivative_filter(ecg):
    # Make impulse response
    h = [-1, -2, 0, 2, 1]
    h = [x / 8 for x in h]
    # Apply filter
    ecg4 = np.convolve(ecg, h)
    ecg4 = np.roll(ecg4, -6)
    return normalize_ecg(ecg4)


def squaring(ecg):
    ecg5 = np.square(ecg)
    return normalize_ecg(ecg5)


def moving_window_integration(ecg):
    # Make impulse response
    h = np.ones(31)
    h = np.array([x / 31 for x in h])

    # Apply filter
    ecg6 = np.convolve(ecg, h)
    ecg6 = np.roll(ecg6, -15)
    return normalize_ecg(ecg6)


def left_right(ecg6):
    max_h = max(ecg6)
    thresh = np.mean(ecg6)
    poss_reg = np.transpose(apply(ecg6, lambda x: x > max_h * thresh))

    left_reg = np.append([0], poss_reg)
    left_diff = diff(left_reg)
    left = find(left_diff, lambda x: x == 1)

    right_reg = np.append(poss_reg, [0])
    right_diff = diff(right_reg)
    right = find(right_diff, lambda x: x == -1)

    return left, right


def qrs(ecg1, left, right):
    R_values = []
    R_locs = []
    for i in range(len(left)):
        if left[i] >= right[i] or left[i] < 0 or right[i] > len(ecg1):
            # print('Ignoring range', left[i], right[i])
            continue

        R_value, R_loc = np_max(ecg1[left[i]:right[i]])

        if R_loc == 0 or R_loc == right[i] - left[i]:
            # print('Ignoring range', left[i], right[i], 'R_loc is at the edge')
            continue

        R_loc = R_loc + left[i]
        R_values.append(R_value)
        R_locs.append(R_loc)

    R_locs = [R_locs[i] for i in find(R_locs, lambda x: x != 0)]
    return R_locs
