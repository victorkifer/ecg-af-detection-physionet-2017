import csv
import scipy.io as sio
import preprocessing

# Default dir where data set is stored
__DATA_DIR = '/usr/share/ml_data_sets/CinC_ECG/training2017'


def load_all_data(data_path=__DATA_DIR):
    (data, labels) = __load_data(data_path)
    return preprocessing.shuffle_data(data, labels)


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
