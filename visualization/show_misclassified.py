import csv
from collections import Counter

import matplotlib.pyplot as plt
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
        else:
            acc[true_label] = acc.get(true_label, 0) + 1

    total = Counter(ytrue)

    f_score = []
    for key in total.keys():
        f = acc.get(key, 0) / total[key]
        print(key, f)
        f_score.append(f)

    print('Final score', sum(f_score) / len(f_score))

    while (True):
        name = input("Enter an entry name to plot: ")
        if len(name.strip()) == 0:
            break

        row = loader.load_data_from_file(name)
        r = qrs_detect(normalize_ecg(remove_dc_component(row)))
        plt.plot(range(len(row)), row, 'g-',
                 r, [row[x] for x in r], 'r^')
        plt.show()
