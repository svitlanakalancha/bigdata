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
