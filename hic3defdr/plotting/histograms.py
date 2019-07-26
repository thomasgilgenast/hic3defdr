import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter


@plotter
def plot_pvalue_histogram(data, xlabel='pvalue', **kwargs):
    """
    Plots a p-value or q-value distribution.

    Parameters
    ----------
    data : np.ndarray
        The p-values or q-values to plot.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    plt.hist(data, bins=np.linspace(0, 1, 21))
    plt.ylabel('number of pixels')
