import random
import numpy as np
import matplotlib.pyplot as plt
from pywt import dwt, wavedec

from qrs_detect import r_detect

plt.rcParams["figure.figsize"] = (20, 6)

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


def plot_wavelet(row, clazz):
    a, d1, d2 = wavedec(row, 'db1', level=2)

    plt.title(clazz)
    plt.subplot(4, 1, 1)
    plt.plot(range(len(row)), row, 'g-')
    plt.ylabel("ECG")

    plt.subplot(4, 1, 2)
    plt.plot(range(len(a)), a, 'g-')
    plt.ylabel("Wavelet")

    plt.subplot(4, 1, 3)
    plt.plot(range(len(d1)), d1, 'g-')
    plt.ylabel("D1")

    plt.subplot(4, 1, 4)
    plt.plot(range(len(d2)), d2, 'g-')
    plt.ylabel("D2")

    print(clazz)
    print('Mean', np.mean(d1))
    print('Std', np.std(d1))
    print('Mean', np.mean(d2))
    print('Std', np.std(d2))

    plt.show()


# Normal: A00001, A00002, A0003, A00006
plot_wavelet(loader.load_data_from_file("A00001"), "Normal")
# AF: A00004, A00009, A00015, A00027
plot_wavelet(loader.load_data_from_file("A00004"), "AF rhythm")
# Other: A00005, A00008, A00013, A00017
plot_wavelet(loader.load_data_from_file("A00005"), "Other rhythm")
# Noisy: A00205, A00585, A01006, A01070
plot_wavelet(loader.load_data_from_file("A00205"), "Noisy signal")
