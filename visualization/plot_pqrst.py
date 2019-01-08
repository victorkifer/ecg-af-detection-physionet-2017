"""
Plots records and detected R peaks
applying both QRS detection algorithms

See common/qrs_detect.py, common/qrs_detect2.py
"""

import random

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from features.qrs_detect import *

plt.rcParams["figure.figsize"] = (12, 4)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

from loading import loader

from features.qrs_detect2 import *


def plot_with_detected_peaks(row, clazz):
    row = row[12 * 300:15 * 300]
    row = normalizer.normalize_ecg(row)
    p, q, r, s, t = pqrst_detect(row)

    plt.grid(alpha=0.5)
    plt.xticks(np.arange(0, 10.1, 0.5))
    plt.yticks(np.arange(-1, 1.01, 0.2))
    plt.plot([x / 300 for x in range(len(row))], row, 'k-',
             [x / 300 for x in p], [row[x] for x in p], 'go',
             [x / 300 for x in q], [row[x] for x in q], 'bv',
             [x / 300 for x in r], [row[x] for x in r], 'r^',
             [x / 300 for x in s], [row[x] for x in s], 'bv',
             [x / 300 for x in t], [row[x] for x in t], 'mo',
             )
    plt.title("Positions of P, Q, R, S, T")
    plt.xlabel("Time, s")
    plt.ylabel("Normalized ECG signal")

    plt.show()


# Normal: A00001, A00002, A0003, A00006
# AF: A00004, A00009, A00015, A00027
# Other: A00005, A00008, A00013, A00017
# Noisy: A00205, A00585, A01006, A01070
plot_with_detected_peaks(loader.load_data_from_file("A00001"), "Normal")
