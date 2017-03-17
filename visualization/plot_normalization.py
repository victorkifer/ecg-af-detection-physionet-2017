import random
import numpy as np
import matplotlib.pyplot as plt

from qrs_detect import *

plt.rcParams["figure.figsize"] = (20, 4)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

import loader
import preprocessing

(X, Y) = loader.load_all_data()
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)


def plot(ecg):
    plt.plot(ecg)
    plt.show()


def plot2(ecg, ecg1):
    l1, = plt.plot(ecg, label="Line 1")
    l2, = plt.plot(ecg1, label="Line 2")
    plt.grid()
    plt.legend(handles=[l1, l2])
    plt.show()


def plot_with_detected_peaks(ecg):
    # plot(ecg)
    ecg1 = preprocessing.normalize_ecg(ecg)
    print(len(ecg), "->", len(ecg1))
    # plot(ecg1)
    ecg2 = low_pass_filtering(ecg1)
    print(len(ecg1), "->", len(ecg2))
    # plot2(ecg1, ecg2)
    # plot(ecg2)
    ecg3 = high_pass_filtering(ecg2)
    print(len(ecg2), "->", len(ecg3))
    # plot2(ecg2, ecg3)
    # plot(ecg3)
    ecg4 = derivative_filter(ecg3)
    print(len(ecg3), "->", len(ecg4))
    # plot2(ecg3, ecg4)
    # plot(ecg4)
    ecg5 = squaring(ecg4)
    print(len(ecg4), "->", len(ecg5))
    # plot2(ecg4, ecg5)
    # plot(ecg5)
    ecg6 = moving_window_integration(ecg5)
    print(len(ecg5), "->", len(ecg6))
    # plot2(ecg5, ecg6)
    # plot(ecg6)


plot_with_detected_peaks(X[0])
plot_with_detected_peaks(X[1])
plot_with_detected_peaks(X[2])
plot_with_detected_peaks(X[3])
