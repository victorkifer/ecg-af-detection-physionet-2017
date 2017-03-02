import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 4)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

import loader
import preprocessing
import feature_extractor

(X, Y) = loader.load_all_data()
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)


def plot_with_detected_peaks(row):
    peaks = feature_extractor.get_r_peaks_positions(row)

    plt.plot(range(len(row)), row, 'g-', peaks, [row[x] for x in peaks], 'r+')
    plt.show()


plot_with_detected_peaks(X[0])
plot_with_detected_peaks(X[1])
plot_with_detected_peaks(X[2])
plot_with_detected_peaks(X[3])
