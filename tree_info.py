def plot_tree_info(rf):
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib is not installed")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(importances)), indices)
    plt.xlim([-1, len(importances)])
    plt.show()
