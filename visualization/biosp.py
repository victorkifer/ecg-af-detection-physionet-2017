"""
This script plots the wavelet of the signal with details
"""

import random

import matplotlib

import preprocessing
from utils import matlab

matplotlib.use("Qt5Agg")

from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20, 6)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

import loader


def plot_wavelet(row):
    row, transposed = preprocessing.transpose_if_needed(row)
    [ts, fts, rpeaks, tts, thb, hrts, hr] = ecg.ecg(signal=row, sampling_rate=loader.FREQUENCY, show=True)

    rri = np.diff(rpeaks)
    rr_mean = np.mean(rri)

    misclassified = matlab.find(rri, lambda x: x < 0.5 * rr_mean)
    print(misclassified)
    rpeaks = np.delete(rpeaks, misclassified)
    thb = np.delete(thb, misclassified)


# Normal: A00001, A00002, A0003, A00006
plot_wavelet(loader.load_data_from_file("A00002"))
