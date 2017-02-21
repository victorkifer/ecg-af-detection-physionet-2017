import csv
import numpy as np
from random import shuffle
import scipy.io as sio


# Global variables declaration
__DATA_DIR='/usr/share/ml_data_sets/CinC_ECG/training2017'


def load_all_data():
    (data, labels) = __load_data()
    return __shuffle_data(data, labels)
    

def __get_data_from_file(case_name):
    test = sio.loadmat(__DATA_DIR + '/' + case_name + '.mat')
    content = test['val']
    return content


def __load_data():
    data=[]
    labels=[]
    with open(__DATA_DIR + '/REFERENCE.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            file_name = row[0]
            label = row[1]
            data.append(__get_data_from_file(file_name))
            labels.append(label)

    return (data, labels)


def __shuffle_data(data, labels):
    data_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(data)))
    shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        labels_shuf.append(labels[i])
    return (data_shuf, labels_shuf)