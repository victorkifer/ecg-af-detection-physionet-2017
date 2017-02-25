def print_categorical_validation(model, valX, valY, mapping):
    correct = [x.tolist().index(max(x)) for x in valY]
    predicted = [x.tolist().index(max(x)) for x in model.predict(valX)]

    values = [correct[i] == predicted[i] for i in range(len(correct))]
    accuracy = values.count(True) * 1.0 / len(correct)

    matrix_size = len(mapping.keys())

    import numpy as np
    val = np.zeros((matrix_size, matrix_size), np.int32)
    for i in range(len(correct)):
        c = correct[i]
        p = predicted[i]
        val[c][p] += 1

    print("-" * 30)
    print('Overal accuracy', accuracy)
    for i in range(matrix_size):
        classified = val[i][i]
        total = max(sum(val[i]), 1)
        print(mapping[i], 'accuracy is', classified / total)

    print("-" * 30)
    print(val)
    print("-" * 30)
