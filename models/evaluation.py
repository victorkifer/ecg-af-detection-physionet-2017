from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def print_validation_info(trueY, predY):
    print(classification_report(trueY, predY))

    print('Confusion matrix:')
    print(confusion_matrix(trueY, predY))