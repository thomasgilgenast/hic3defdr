import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter

from fast3defdr.scaled_nb import mvr
from fast3defdr.dispersion import mme_per_pixel


@plotter
def plot_variance_fit(mean, var, disp, mean_per_bin, disp_per_bin, **kwargs):
    """
    Plots mean versus pixel-wise, bin-wise, and smoothed dispersion in terms of
    variance.

    Parameters
    ----------
    mean, var : np.ndarray
        The pixel-wise mean and variance, respectively.
    disp : np.ndarray
        The smoothed dispersion estimate for each pixel.
    mean_per_bin, disp_per_bin : np.ndarray
        The mean and estimated dispersion, respectively, for each bin.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # filter out zeros
    pos_idx = (mean > 0) & (var > 0)
    mean = mean[pos_idx]
    var = var[pos_idx]
    disp = disp[pos_idx]

    # determine reasonable plot limits
    xmin = 1
    xmax = np.percentile(mean, 99.9)
    ymin = np.percentile(var, 0.00001)
    ymax = np.percentile(var, 99.99999)

    # plot
    plt.hexbin(mean, var, bins='log', xscale='log', yscale='log', cmap='Blues',
               extent=np.log10([xmin, xmax, ymin, ymax]))
    plt.scatter(mean_per_bin, disp_per_bin, label=r'$\hat{\sigma^2}$ per bin',
                color='C1')
    sort_idx = np.argsort(mean)[::len(mean)/100]
    plt.plot(mean[sort_idx], mvr(mean[sort_idx], disp[sort_idx]),
             label=r'smoothed $\hat{\sigma^2}$', color='C2')
    plt.plot([xmin, xmax], [xmin, xmax], label='Poisson', ls='--', color='C3')
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.xlabel('mean')
    plt.ylabel('variance')
    plt.legend(loc='lower right')


@plotter
def plot_dispersion_fit(mean, var, disp, mean_per_bin, disp_per_bin, **kwargs):
    """
    Plots mean versus pixel-wise, bin-wise, and smoothed dispersion in terms of
    dispersion.

    Parameters
    ----------
    mean, var : np.ndarray
        The pixel-wise mean and variance, respectively.
    disp : np.ndarray
        The smoothed dispersion estimate for each pixel.
    mean_per_bin, disp_per_bin : np.ndarray
        The mean and estimated dispersion, respectively, for each bin.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # filter out zeros
    pos_idx = (mean > 0)
    mean = mean[pos_idx]
    var = var[pos_idx]
    disp = disp[pos_idx]

    # determine reasonable plot limits
    xmin = 1
    xmax = np.percentile(mean, 99.9)
    ymin = -1
    ymax = 2

    # compute disp per pixel
    disp_per_pixel = mme_per_pixel(mean, var)

    # plot
    plt.hexbin(mean, disp_per_pixel, bins='log', xscale='log', cmap='Blues',
               extent=(np.log10(xmin), np.log10(xmax), ymin, ymax))
    plt.scatter(mean_per_bin, disp_per_bin, label=r'$\hat{\alpha}$ per bin',
                color='C1')
    sort_idx = np.argsort(mean)[::len(mean) / 100]
    plt.plot(mean[sort_idx], disp[sort_idx], label=r'smoothed $\hat{\alpha}$',
             color='C2')
    plt.hlines(0, xmin, xmax, label='Poisson', ls='--', color='C3')
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    plt.plot(xs, [mme_per_pixel([[x, x]]) for x in xs], ls='--', color='gray')
    plt.plot(xs, [mme_per_pixel([[0, 2*x]]) for x in xs], ls='--', color='gray')
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.xlabel('mean')
    plt.ylabel('dispersion')
    plt.legend(loc='lower right')
