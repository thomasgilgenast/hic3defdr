import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter
from lib5c.plotters.scatter import scatter

from fast3defdr.scaled_nb import mvr, inverse_mvr
from fast3defdr.dispersion import mme_per_pixel


@plotter
def plot_variance_fit(mean, var, disp, cov_per_bin, disp_per_bin, dist=None,
                      dist_max=200, add_curve=True, **kwargs):
    """
    Plots pixel-wise, bin-wise, and smoothed dispersion in terms of variance
    versus either pixel-wise mean or distance.

    Parameters
    ----------
    mean, var : np.ndarray
        The pixel-wise mean and variance, respectively.
    disp : np.ndarray
        The smoothed dispersion estimate for each pixel.
    cov_per_bin, disp_per_bin : np.ndarray
        The covariate and estimated dispersion, respectively, for each bin.
    dist : np.ndarray, optional
        Pass the distance per-pixel to plot the variance against distance. Pass
        None to plot the variance against pixel-wise mean instead.
    dist_max : int
        If plotting variance versus distance, the maximum distance to plot in
        bin units.
    add_curve : bool
        Pass True to include the curve of smoothed dispersions. Pass False to
        skip it.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # filter out zeros
    if dist is not None:
        pos_idx = (mean > 0) & (var > 0) & (dist <= dist_max)
    else:
        pos_idx = (mean > 0) & (var > 0)
    mean = mean[pos_idx]
    var = var[pos_idx]
    disp = disp[pos_idx]
    if dist is not None:
        dist = dist[pos_idx]
        cov = dist
        xlabel = 'distance'
        logx = False
    else:
        cov = mean
        xlabel = 'mean'
        logx = True

    # determine reasonable plot limits
    if dist is None:
        xmin = 5  # 1
        xmax = np.percentile(mean, 99.99)
    else:
        xmin = 0
        xmax = dist_max
    ymin = np.percentile(var, 1)
    ymax = np.percentile(var, 99)

    # compute mean and var per bin
    if dist is not None:
        dd = np.array([np.mean(mean[dist == d]) for d in range(dist_max + 1)])
        mean_per_bin = np.array([dd[int(np.rint(c))] if c <= dist_max else 0
                                 for c in cov_per_bin])
    else:
        mean_per_bin = cov_per_bin
        dd = None
    var_per_bin = mvr(mean_per_bin, disp_per_bin)

    # plot
    scatter(cov, var, logx=logx, logy=True, hexbin=True, xlim=[xmin, xmax],
            ylim=[ymin, ymax])
    plt.scatter(cov_per_bin, var_per_bin, label=r'$\hat{\sigma}^2$ per bin',
                color='C1')
    if add_curve:
        sort_idx = np.argsort(cov)[::len(cov)/1000]
        plt.plot(cov[sort_idx], mvr(dd[cov[sort_idx]], disp[sort_idx]),
                 label=r'smoothed $\hat{\sigma}^2$', linewidth=3, color='C4')
    if dist is None:
        plt.plot([xmin, xmax], [xmin, xmax], label='Poisson', linestyle='--',
                 linewidth=3, color='C3')
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.xlabel(xlabel)
    plt.ylabel('variance')
    plt.legend(loc='lower right')


@plotter
def plot_dispersion_fit(mean, var, disp, cov_per_bin, disp_per_bin, dist=None,
                        dist_max=200, add_curve=True, **kwargs):
    """
    Plots mean versus pixel-wise, bin-wise, and smoothed dispersion in terms of
    dispersion.

    Parameters
    ----------
    mean, var : np.ndarray
        The pixel-wise mean and variance, respectively.
    disp : np.ndarray
        The smoothed dispersion estimate for each pixel.
    cov_per_bin, disp_per_bin : np.ndarray
        The covariate and estimated dispersion, respectively, for each bin.
    dist : np.ndarray, optional
        Pass the distance per-pixel to plot the variance against distance. Pass
        None to plot the variance against pixel-wise mean instead.
    dist_max : int
        If plotting variance versus distance, the maximum distance to plot in
        bin units.
    add_curve : bool
        Pass True to include the curve of smoothed dispersions. Pass False to
        skip it.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # filter out zeros
    if dist is not None:
        pos_idx = (mean > 0) & (var > 0) & (dist <= dist_max)
    else:
        pos_idx = (mean > 0) & (var > 0)
    mean = mean[pos_idx]
    var = var[pos_idx]
    disp = disp[pos_idx]
    if dist is not None:
        dist = dist[pos_idx]
        cov = dist
        xlabel = 'distance'
        logx = False
    else:
        cov = mean
        xlabel = 'mean'
        logx = True

    # compute disp per pixel
    disp_per_pixel = inverse_mvr(mean, var)

    # determine reasonable plot limits
    if dist is None:
        xmin = 5  # 1
        xmax = np.percentile(mean, 99.99)
    else:
        xmin = 0
        xmax = dist_max
    ymin = np.percentile(disp_per_pixel, 1)
    ymax = np.percentile(disp_per_pixel, 99)

    # plot
    scatter(cov, disp_per_pixel, logx=logx, logy=False, hexbin=True,
            xlim=[xmin, xmax], ylim=[ymin, ymax])
    plt.scatter(cov_per_bin, disp_per_bin, label=r'$\hat{\alpha}$ per bin',
                color='C1')
    if add_curve:
        sort_idx = np.argsort(cov)[::len(cov)/1000]
        plt.plot(cov[sort_idx], disp[sort_idx],
                 label=r'smoothed $\hat{\alpha}$', linewidth=3, color='C4')
    plt.hlines(0, xmin, xmax, label='Poisson', linestyle='--', linewidth=3,
               color='C3')
    if dist is None:
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 100)
        plt.plot(xs, [mme_per_pixel([[x, x]]) for x in xs], linestyle='--',
                 linewidth=3, color='gray')
        plt.plot(xs, [mme_per_pixel([[0, 2*x]]) for x in xs], linestyle='--',
                 linewidth=3, color='gray')
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.xlabel(xlabel)
    plt.ylabel('dispersion')
    plt.legend(loc='upper right')
