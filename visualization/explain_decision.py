"""
Reads answers.txt file and REFERENCE.csv file
and compares correct labes with predicted
Than outputs the list of wrongly classified training samples

NOTE:
    this script provides you an ability to plot wrongly classified entries
NOTE:
    make sure you have generated the answers.txt file
"""

import matplotlib
import sklearn
from sklearn.externals import joblib

import feature_extractor5
import loader
import preprocessing

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

import numpy as np

model_file = "../model.pkl"

name = input("Enter an entry name to plot: ")
name = name.strip()
if len(name) == 0:
    print("Finishing")
    exit(-1)

if not loader.check_has_example(name):
    print("File Not Found")
    exit(-1)

x = loader.load_data_from_file(name, "../validation")
x = preprocessing.normalize_ecg(x)
x = feature_extractor5.features_for_row(x)

# as we have one sample at a time to predict, we should resample it into 2d array to classify
x = np.array(x).reshape(1, -1)

model = joblib.load(model_file)
tree = model

# import pydotplus
# dot_data = sklearn.tree.export_graphviz(tree, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")


def explain(tree, x):
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    node_indicator = tree.decision_path(x)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    print(x)

    for node_id in node_indicator.indices:
        if (x[0, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
              % (node_id,
                 0,
                 feature[node_id],
                 x[0, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))


def get_code(tree, feature_names = None, tabdepth=0):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    if feature_names is None:
        features = ['f%d' % i for i in tree.tree_.feature]
    else:
        features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, tabdepth=0):
            if (threshold[node] != -2):
                    print('\t' * tabdepth,
                          "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                    if left[node] != -1:
                            recurse (left, right, threshold, features,left[node], tabdepth+1)
                    print('\t' * tabdepth,
                          "} else {")
                    if right[node] != -1:
                            recurse (left, right, threshold, features,right[node], tabdepth+1)
                    print('\t' * tabdepth,
                          "}")
            else:
                    print('\t' * tabdepth,
                          "return " + str(value[node]))

    recurse(left, right, threshold, features, 0)



# get_code(tree)

explain(tree, x)