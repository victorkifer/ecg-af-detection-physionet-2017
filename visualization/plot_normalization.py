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


def plot_with_detected_peaks(ecg):
    plot(ecg)
    ecg1 = cancel_dc_drift(ecg)
    plot(ecg1)
    ecg2 = low_pass_filtering(ecg1)
    plot(ecg2)
    ecg3 = high_pass_filtering(ecg2)
    plot(ecg3)
    ecg4 = derivative_filter(ecg3)
    plot(ecg4)
    ecg5 = squaring(ecg4)
    plot(ecg5)
    ecg6 = moving_window_integration(ecg5)
    plot(ecg6)


plot_with_detected_peaks(X[0])
plot_with_detected_peaks(X[1])
plot_with_detected_peaks(X[2])
plot_with_detected_peaks(X[3])
