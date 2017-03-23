import random

import matplotlib.pyplot as plt

import loader
from common.qrs_detect import *

plt.rcParams["figure.figsize"] = (20, 8)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)


def plot_with_detected_peaks(ecg, clazz):
    total = 7

    plt.subplot(total, 1, 1)
    plt.ylabel(clazz)
    plt.plot(ecg)
    ecg1 = normalize_ecg(remove_dc_component(ecg))

    plt.subplot(total, 1, 2)
    plt.ylabel("DC Normalized")
    plt.plot(ecg1)
    ecg2 = low_pass_filtering(ecg1)

    plt.subplot(total, 1, 3)
    plt.ylabel("LP filter")
    plt.plot(ecg2)
    ecg3 = high_pass_filtering(ecg2)

    plt.subplot(total, 1, 4)
    plt.ylabel("HP filter")
    plt.plot(ecg3)
    ecg4 = derivative_filter(ecg3)

    plt.subplot(total, 1, 5)
    plt.ylabel("Derivative")
    plt.plot(ecg4)
    ecg5 = squaring(ecg4)

    plt.subplot(total, 1, 6)
    plt.ylabel("Square")
    plt.plot(ecg5)
    ecg6 = moving_window_integration(ecg5)

    plt.subplot(total, 1, 7)
    plt.ylabel("Window")
    plt.plot(ecg6)
    plt.show()


# Normal: A00001, A00002, A0003, A00006
plot_with_detected_peaks(loader.load_data_from_file("A00001"), "Normal")
plot_with_detected_peaks(loader.load_data_from_file("A00006"), "Normal")
# AF: A00004, A00009, A00015, A00027
plot_with_detected_peaks(loader.load_data_from_file("A00004"), "AF rhythm")
plot_with_detected_peaks(loader.load_data_from_file("A00009"), "AF rhythm")
# Other: A00005, A00008, A00013, A00017
plot_with_detected_peaks(loader.load_data_from_file("A00005"), "Other rhythm")
plot_with_detected_peaks(loader.load_data_from_file("A00008"), "Other rhythm")
# Noisy: A00205, A00585, A01006, A01070
plot_with_detected_peaks(loader.load_data_from_file("A00205"), "Noisy signal")
plot_with_detected_peaks(loader.load_data_from_file("A00585"), "Noisy signal")
