import csv
from random import shuffle
import scipy.io as sio

# Default dir where data set is stored
__DATA_DIR = '/usr/share/ml_data_sets/CinC_ECG/training2017'


def load_all_data(data_path=__DATA_DIR):
    (data, labels) = __load_data(data_path)
    return __shuffle_data(data, labels)


def __get_data_from_file(data_path, example_name):
    """
    Loads data from MatLab file for given example
    :return: features for given example
    """
    test = sio.loadmat(data_path + '/' + example_name + '.mat')
    content = test['val'][0]
    return content


def __load_data(data_path):
    data = []
    labels = []
    with open(data_path + '/REFERENCE.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            file_name = row[0]
            label = row[1]
            data.append(__get_data_from_file(data_path, file_name))
            labels.append(label)

    return (data, labels)


def __shuffle_data(data, labels):
    """
    Shuffles input data

    In some cases input data might be distributed sorted which might create a hidden error
    in training/validation process so it's better to always shuffle input data before usage
    :return: Shuffled input data
    """
    data_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(data)))
    shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        labels_shuf.append(labels[i])
    return (data_shuf, labels_shuf)
