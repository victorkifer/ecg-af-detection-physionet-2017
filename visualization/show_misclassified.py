import csv
from collections import Counter

with open('../answers.txt') as predicted, \
        open('../validation/REFERENCE.csv') as correct:
    preader = csv.reader(predicted)
    creader = csv.reader(correct)

    ypred = []
    ytrue = []

    acc = dict()

    print("Record Pred Corr")
    for (p, c) in zip(preader, creader):
        (record, pred_label) = p
        true_label = c[1]
        ypred.append(pred_label)
        ytrue.append(true_label)

        if p != c:
            print(record, pred_label, true_label)
        else:
            acc[true_label] = acc.get(true_label, 0) + 1

    total = Counter(ytrue)

    f_score = []
    for key in total.keys():
        f = acc.get(key, 0) / total[key]
        print(key, f)
        f_score.append(f)

    print('Final score', sum(f_score) / len(f_score))
