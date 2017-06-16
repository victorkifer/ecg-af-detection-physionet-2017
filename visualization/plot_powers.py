import matplotlib
from common.qrs_detect import low_pass_filtering, high_pass_filtering
from scipy import signal

import preprocessing
from features import hrv
from loading import loader
from preprocessing import normalizer

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 3)


def plot_powers(x):
    x = normalizer.normalize_ecg(x)
    x = low_pass_filtering(x)
    x = high_pass_filtering(x)
    fs, powers = signal.welch(x, loader.FREQUENCY)

    fs = fs[:50]
    powers = powers[:50]

    print(hrv.frequency_domain(x, fs=300))

    plt.plot(fs, powers)
    plt.xticks([2 * i for i in range(len(fs) // 2)])
    plt.grid()
    plt.show()


plot_powers(loader.load_data_from_file("A00001"))
plot_powers(loader.load_data_from_file("A00004"))
plot_powers(loader.load_data_from_file("A00013"))
plot_powers(loader.load_data_from_file("A01006"))
