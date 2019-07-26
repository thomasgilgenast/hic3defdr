import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter
from lib5c.plotters.scatter import scatter

from hic3defdr.scaled_nb import mvr, inverse_mvr


@plotter
def plot_mvr(mean, var, disp=None, cov_per_bin=None, disp_per_bin=None,
             dist=None, dist_max=200, yaxis='var', mean_thresh=5.0,
             scatter_fit=-1, scatter_size=36, logx=True, logy=True, ylim=None,
             xlim=None, legend=True, **kwargs):
    """
    Plots pixel-wise, bin-wise, and estimated dispersions in terms of either
    dispersion or variance versus either pixel-wise mean or distance.

    Parameters
    ----------
    mean, var : np.ndarray
        The pixel-wise mean and variance, respectively.
    disp : np.ndarray, optional
        The dispersion estimate for each pixel. Pass None to omit.
    cov_per_bin, disp_per_bin : np.ndarray, optional
        The covariate and estimated dispersion, respectively, for each bin.
        Pass None to omit.
    dist : np.ndarray, optional
        Pass the distance per-pixel to plot the variance against distance. Pass
        None to plot the variance against pixel-wise mean instead.
    dist_max : int
        If plotting variance versus distance, the maximum distance to plot in
        bin units.
    yaxis : 'var' or 'disp'
        What to plot on the y-axis.
    mean_thresh : float
        Pass the minimum mean threshold used during dispersion estimation to
        set the left x-axis limit when plotting mean on the x-axis.
    scatter_fit : int
        Pass a nonzero integer to draw the fitted dispersions passed in
        ``disp`` as a scatterplot of ``scatter_fit`` selected points. Pass -1
        to plot the fitted dispersions passed in ``disp`` as a curve. Pass 0 to
        omit plotting the dispersion estimates passed in ``disp`` altogether.
    scatter_size : int
        The marker size when plotting scatterplots.
    logx, logy : bool
        Whether or not to log the x- or y-axis, respectively.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # ensure y
    if yaxis == 'var':
        y = var
        yname = r'$\hat{\sigma}^2$'
        ylabel = 'variance'
    elif yaxis == 'disp':
        y = inverse_mvr(mean, var)
        yname = r'$\hat{\alpha}$'
        ylabel = 'dispersion'
    else:
        raise ValueError('yaxis must be \'disp\' or \'var\'')

    # determine reasonable plot limits
    if dist is None:
        xmin = mean_thresh
        xmax = np.percentile(mean, 99.99)
    else:
        xmin = 0
        xmax = dist_max
    ymin = max(np.percentile(y, 0.1), 1e-7) if logy \
        else min(np.percentile(y, 0.1), 0)
    ymax = np.percentile(y, 99.9)

    # override plot limits
    if xlim is not None:
        xmin, xmax = xlim
    if ylim is not None:
        ymin, ymax = ylim

    # filter out zeros
    if dist is not None:
        pos_idx_1 = (mean > 0) & (var > 0) & (dist <= dist_max)
    else:
        pos_idx_1 = (mean > 0) & (var > 0)
    if logy:
        pos_idx_2 = y > 0
    else:
        pos_idx_2 = np.ones_like(pos_idx_1)
    pos_idx = pos_idx_1 & pos_idx_2
    mean = mean[pos_idx]
    var = var[pos_idx]
    y = y[pos_idx]
    disp = disp[pos_idx] if disp is not None else None
    if dist is not None:
        dist = dist[pos_idx]
        x = dist
        xlabel = 'distance'
    else:
        x = mean
        xlabel = 'mean'

    # check for bins
    bins = cov_per_bin is not None and disp_per_bin is not None

    if bins:
        # ensure mean per bin
        if dist is not None:
            dd = np.array([np.mean(mean[dist == d])
                           for d in range(dist_max + 1)])
            mean_per_bin = np.array([dd[int(np.rint(c))]
                                     if c <= dist_max else 0
                                     for c in cov_per_bin])
        else:
            mean_per_bin = cov_per_bin
            dd = None

        # ensure y per bin
        if yaxis == 'var':
            y_per_bin = mvr(mean_per_bin, disp_per_bin)
        elif yaxis == 'disp':
            y_per_bin = disp_per_bin
        else:
            raise ValueError('yaxis must be \'disp\' or \'var\'')
    else:
        y_per_bin = None

    # hexbin per-pixel values
    scatter(x, y, logx=logx, logy=logy, hexbin=True, xlim=[xmin, xmax],
            ylim=[ymin, ymax])

    # scatter per-bin estimates
    if bins:
        plt.scatter(cov_per_bin, y_per_bin, label='%s per bin' % yname,
                    color='C1', s=scatter_size)

    # plot or scatter fitted estimates
    if disp is not None:
        # step 1: prepare sort indices
        sort_idx = np.argsort(x)
        if dist is not None:
            mean_sort_idx = dd[x[sort_idx]]
        else:
            mean_sort_idx = x[sort_idx]

        # step 2: ensure y_hat_sorted
        if yaxis == 'var':
            y_hat_sorted = mvr(mean_sort_idx, disp[sort_idx])
        elif yaxis == 'disp':
            y_hat_sorted = disp[sort_idx]
        else:
            raise ValueError('yaxis must be \'disp\' or \'var\'')

        # step 3: plot or scatter or neither
        if scatter_fit > 0:
            plt.scatter(x[sort_idx][::len(x)/scatter_fit],
                        y_hat_sorted[::len(x)/scatter_fit], label=yname,
                        color='C4', s=scatter_size)
        elif scatter_fit == -1:
            plt.plot(x[sort_idx], y_hat_sorted, label='smoothed %s' % yname,
                     linewidth=3, color='C4')

    # add poisson line
    if dist is None:
        if yaxis == 'var':
            xs = 10**np.linspace(np.log10(xmin), np.log10(xmax), 100) if logx \
                else np.linspace(xmin, xmax, 100)
            plt.plot(xs, xs, label='Poisson', linestyle='--', linewidth=3,
                     color='C3')
        elif yaxis == 'disp':
            plt.hlines(0, xmin, xmax, label='Poisson', linestyle='--',
                       linewidth=3, color='C3')
        else:
            raise ValueError('yaxis must be \'disp\' or \'var\'')

    # cleanup
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
