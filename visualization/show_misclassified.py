"""
Reads answers.txt file and REFERENCE.csv file
and compares correct labes with predicted
Than outputs the list of wrongly classified training samples

NOTE:
    this script provides you an ability to plot wrongly classified entries
NOTE:
    make sure you have generated the answers.txt file
"""

import csv
from collections import Counter

import matplotlib
matplotlib.use("Qt5Agg")

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
    info = dict()

    misclassified = []

    for (p, c) in zip(preader, creader):
        (record, pred_label) = p
        true_label = c[1]
        ypred.append(pred_label)
        ytrue.append(true_label)

        if p != c:
            info[record] = "Pred=" + pred_label + " True=" + true_label
            misclassified.append((true_label, pred_label, record))
        else:
            info[record] = "Correct=" + true_label

    misclassified = sorted(misclassified, key=lambda t: t[0] + t[1])
    for item in misclassified:
        print(item[2], 'is of class', item[0], 'but was classified as', item[1])

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
        name = name.strip()
        if len(name) == 0:
            print("Finishing")
            break

        if not loader.check_has_example(name):
            print("File Not Found")
            continue

        row = loader.load_data_from_file(name)
        r = qrs_detect(normalize_ecg(remove_dc_component(row)))
        plt.plot(range(len(row)), row, 'g-',
                 r, [row[x] for x in r], 'r^')
        plt.ylabel(info[name])
        plt.show()
