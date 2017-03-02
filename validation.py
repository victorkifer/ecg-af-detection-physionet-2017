from sklearn.metrics import confusion_matrix, accuracy_score


def extract_validation(trueY, predY, categorical=False):
    correct = from_categorical(trueY) if categorical else trueY
    predicted = from_categorical(predY) if categorical else predY

    matrix = confusion_matrix(correct, predicted)
    accuracy = accuracy_score(correct, predicted)

    accuracies = []
    for i in range(matrix.shape[0]):
        total = max(1, sum(matrix[i]))
        accuracies.append(matrix[i][i] / total)

    return (accuracy, accuracies, matrix)


def print_categorical_validation(trueY, predY, categorical=False):
    (accuracy, accuracies, matrix) = extract_validation(trueY, predY, categorical)

    print('Confusion matrix:')
    print(matrix)
    for i in range(len(accuracies)):
        print('Accuracy for', i, 'is', accuracies[i])

    print('Accuracy', accuracy)


def from_categorical(y):
    return [x.tolist().index(max(x)) for x in y]
