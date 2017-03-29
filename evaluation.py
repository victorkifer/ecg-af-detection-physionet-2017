from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def print_validation_info(trueY, predY, categorical=False):
    trueY = from_categorical(trueY) if categorical else trueY
    predY = from_categorical(predY) if categorical else predY

    print(classification_report(trueY, predY))

    print('Confusion matrix:')
    print(confusion_matrix(trueY, predY))


def from_categorical(y):
    return [x.tolist().index(max(x)) for x in y]
