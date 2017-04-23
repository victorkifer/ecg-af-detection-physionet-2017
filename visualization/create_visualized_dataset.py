import csv
from functools import partial
from os import path

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from pywt import wavedec

import loader
import preprocessing
from common import qrs_detect
from utils import async
from utils import logger
from utils.system import mkdir

plt.rcParams["figure.figsize"] = (40, 3)

logger.enable_logging("ds")


def __draw_to_file(file, values):
    plt.plot(values)
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.close()


def extract(record, data_dir):
    x = loader.load_data_from_file(record, data_dir)
    x = qrs_detect.remove_dc_component(x)
    x = qrs_detect.normalize_ecg(x)

    a = preprocessing.numpy_set_length(x, 20000)
    a, d1, d2, d3 = wavedec(a, 'db1', level=3)

    x = qrs_detect.low_pass_filtering(x)
    x = qrs_detect.high_pass_filtering(x)
    x = qrs_detect.derivative_filter(x)
    b = preprocessing.numpy_set_length(x, 20000)
    b, d1, d2, d3 = wavedec(b, 'db1', level=3)

    return (record, a, b)


def draw_all(data_dir, output_raw, output_cleaned):
    with open(data_dir + '/RECORDS', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        names = [row[0] for row in reader]

        print("Starting data extraction...")

        func = partial(extract, data_dir=data_dir)
        results = async.apply_async(names, func)

        print("Finished with the data extraction. Plotting...")

        mkdir(output_raw)
        mkdir(output_cleaned)
        for value in results:
            name, a, b = value
            print(name)
            file1 = path.join(output_raw, name + ".png")
            file2 = path.join(output_cleaned, name + ".png")

            # this function does not support multi-threading
            __draw_to_file(file1, a)
            __draw_to_file(file2, b)


if __name__ == "__main__":
    draw_all("../validation", "../outputs/dataset1", "../outputs/dataset2")
