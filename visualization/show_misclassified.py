import csv
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.figsize"] = (20, 6)

import loader
from common.qrs_detect import qrs_detect, normalize_ecg, remove_dc_component

with open('../answers.txt') as predicted, \
        open('../validation/REFERENCE.csv') as correct:
    preader = csv.reader(predicted)
    creader = csv.reader(correct)

    ypred = []
    ytrue = []

    acc = dict()

    for (p, c) in zip(preader, creader):
        (record, pred_label) = p
        true_label = c[1]
        ypred.append(pred_label)
        ytrue.append(true_label)

        if p != c:
            print(record, 'was classified as', pred_label, 'but should be', true_label)

    print(classification_report(ytrue, ypred))

    matrix = confusion_matrix(ytrue, ypred)
    print(matrix)
    for row in matrix:
        amax = sum(row)
        if amax > 0:
            for i in range(len(row)):
                row[i] = row[i] * 100.0 / amax

    print(matrix)

    while (True):
        name = input("Enter an entry name to plot: ")
        if len(name.strip()) == 0:
            break

        row = loader.load_data_from_file(name)
        r = qrs_detect(normalize_ecg(remove_dc_component(row)))
        plt.plot(range(len(row)), row, 'g-',
                 r, [row[x] for x in r], 'r^')
        plt.show()
