import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def visualize_clusters(data, clusters):
    if data.shape[1] > 2:
        raise Error('Too many dimensions')

    if len(data) != len(clusters):
        raise Error('Shape mismatch')

    fig = plt.figure()
    cmap = cm.get_cmap('Spectral')
    unique_clusters = set(clusters)
    dimensions = data.columns

    for label in unique_clusters:
        to_plot = data[clusters == label]
        color = cmap((label + 1) / len(unique_clusters))

        plt.plot(to_plot[dimensions[0]], to_plot[dimensions[1]],
                 '.', color=color)
        plt.title(dimensions[0] + ' vs ' + dimensions[1] + ' clustering')
        plt.xlabel(dimensions[0])
        plt.ylabel(dimensions[1])

    return fig


def visualize_boundaries(clf, data_train, target_train, data_test, target_test):
    if len(target_train) != len(data_train):
        raise Error('Shape mismatch in training data')

    if len(target_test) != len(data_test):
        raise Error('Shape mismatch in testing data')

    if data_train.shape[1] < 2 or data_test.shape[1] < 2:
        raise Error('data must have at least 2 input columns')

    dimensions = data_train.columns

    # create grid
    max_x = max(data_train[dimensions[0]].max(), data_test[dimensions[0]].max())
    min_x = min(data_train[dimensions[0]].min(), data_test[dimensions[0]].min())
    max_y = max(data_train[dimensions[1]].max(), data_test[dimensions[1]].max())
    min_y = min(data_train[dimensions[1]].min(), data_test[dimensions[1]].min())

    x = np.arange(min_x-1, max_x+1, 0.1)
    y = np.arange(min_y-1, max_y+1, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})

    # prdiction based on grid
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # plot and color contour
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].contourf(xx, yy, z, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax[0].scatter(data_train[dimensions[0]], data_train[dimensions[1]],
                      c=target_train, cmap=plt.cm.RdYlBu, marker='.')
    ax[0].set_title(dimensions[0] + ' vs ' + dimensions[1] + ' class boundaries train')
    ax[0].set_xlabel(dimensions[0])
    ax[0].set_ylabel(dimensions[1])

    ax[1].contourf(xx, yy, z, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax[1].scatter(data_test[dimensions[0]], data_test[dimensions[1]],
                      c=target_test, cmap=plt.cm.RdYlBu, marker='.')
    ax[1].set_title(dimensions[0] + ' vs ' + dimensions[1] + ' class boundaries test')
    ax[1].set_xlabel(dimensions[0])
    ax[1].set_ylabel(dimensions[1])

    return fig
