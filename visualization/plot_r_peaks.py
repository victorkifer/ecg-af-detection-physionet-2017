import random
import numpy as np
import matplotlib.pyplot as plt

from qrs_detect import r_detect

plt.rcParams["figure.figsize"] = (20, 4)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

import loader
import preprocessing
import feature_extractor

from qrs_detect2 import *

(X, Y) = loader.load_all_data()
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)


def plot_with_detected_peaks(row):
    r = feature_extractor.get_r_peaks_positions(row)
    # r = qrs_detect2(row, fs=300)

    print('R', r)
    times = np.diff(r)
    print(times)
    print(np.mean(times), np.std(times))
    plt.plot(range(len(row)), row, 'g-',
             r, [row[x] for x in r], 'r^')

    plt.show()

# Normal: A00001, A00002, A0003, A00006
plot_with_detected_peaks(loader.load_data_from_file("A00001"))
# AF: A00004, A00009, A00015, A00027
plot_with_detected_peaks(loader.load_data_from_file("A00004"))
# Other: A00005, A00008, A00013, A00017
plot_with_detected_peaks(loader.load_data_from_file("A00005"))
# Noisy: A00205, A00585, A01006, A01070
plot_with_detected_peaks(loader.load_data_from_file("A00205"))
