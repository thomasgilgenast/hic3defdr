import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter
from lib5c.plotters.scatter import scatter

from hic3defdr.util.scaled_nb import mvr, inverse_mvr


@plotter
def compare_disp_fits(fit_fns, labels, max_dist=200, colors=None,
                      linestyles=None, legend=True, **kwargs):
    # resolve colors
    if colors is None:
        colors = ['C%s' % i for i in range(len(fit_fns))]

    # resolve linestyles
    if linestyles is None:
        linestyles = [None] * len(fit_fns)

    # define x values
    xs = np.arange(max_dist + 1)

    # plot
    for f, l, c, ls in zip(fit_fns, labels, colors, linestyles):
        plt.plot(xs, f(xs), color=c, linestyle=ls, label=l)

    # labels
    plt.xlabel('distance')
    plt.ylabel('dispersion')

    # add legend
    if legend:
        plt.legend()


@plotter
def plot_mvr(pixel_mean, pixel_var=None, pixel_disp=None, pixel_dist=None,
             pixel_var_fit=None, pixel_disp_fit=None, mean_per_bin=None,
             dist_per_bin=None, var_per_bin=None, disp_per_bin=None,
             fit_align_dist=False, xaxis='mean', yaxis='var', dist_max=200,
             mean_min=5.0, scatter_fit=-1, scatter_size=36,
             hexbin=True, logx=True, logy=True, xlim=None, ylim=None,
             legend=True, **kwargs):
    """
    Plots pixel-wise, bin-wise, and estimated dispersions in terms of either
    dispersion or variance versus either pixel-wise mean or distance.

    Parameters
    ----------
    pixel_mean : np.ndarray
        The pixel-wise mean.
    pixel_var, pixel_disp : np.ndarray, optional
        The pixel-wise variance and dispersion. Pass only one of these.
    pixel_dist : np.ndarray, optional
        The pixel-wise distance. Not needed for simple MVR plotting.
    pixel_var_fit, pixel_disp_fit : np.ndarray, optional
        The smoothed pixel-wise variance or dispersion estimates. Pass only one
        of these.
    mean_per_bin, dist_per_bin : np.ndarray, optional
        The mean or distance of each bin used for estimating the dispersions but
        before any smoothing was performed. Pass only one of these.
    var_per_bin, disp_per_bin : np.ndarray, optional
        The estimated variance or dispersion of each bin before any smoothing
        was performed. Pass only one of these.
    fit_align_dist : bool
        Pass True if the var/disp was fitted as a function of distance, in which
        case the fitted vars/disps will be aligned to distance rather than in a
        pixel-wise fashion. Use this to fix "jagged" fit lines caused by
        setting ``xaxis`` to a different value than the one the fitting was
        actually performed against.
    xaxis : 'mean' or 'dist'
        What to plot on the x-axis.
    yaxis : 'var' or 'disp'
        What to plot on the y-axis.
    dist_max : int
        If ``xaxis`` is "dist", the maximum distance to plot in bin units.
    mean_min : float
        If `xaxis` is "mean", the minimum mean to plot.
    scatter_fit : int
        Pass a nonzero integer to draw the fitted vars/disps as a scatter plot
        of ``scatter_fit`` selected points. Pass -1 to plot the fitted
        vars/disps as a curve. Pass 0 to omit plotting the var/disp estimates
        altogether.
    scatter_size : int
        The marker size when plotting scatter plots.
    hexbin : bool
        Pass False to skip plotting the hexbins, leaving only the estimated
        variances or dispersions.
    logx, logy : bool
        Whether or not to log the x- or y-axis, respectively.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # convert pixel_* to [unique_]dist_* quantities part 1: dist_mean (dd)
    # dist_* quantities are aligned to np.arange(pixel_dist.max() + 1)
    # dist_* quantities are only made if pixel_dist is passed
    # it's really nice if dist_mean is monotonic, so we will force that here
    if pixel_dist is not None:
        dist_range = np.arange(pixel_dist.max() + 1)
        smallest_seen = np.inf
        dist_mean_list = []
        for d in dist_range:
            currently_seen = np.mean(pixel_mean[pixel_dist == d])
            if np.isfinite(currently_seen) and currently_seen < smallest_seen:
                smallest_seen = currently_seen
                dist_mean_list.append(currently_seen)
            else:
                dist_mean_list.append(smallest_seen)
        dist_mean = np.array(dist_mean_list)
    else:
        dist_mean = None
        dist_range = None

    # mean_per_bin vs dist_per_bin
    if dist_mean is not None and mean_per_bin is None \
            and dist_per_bin is not None:
        mean_per_bin = np.interp(dist_per_bin, dist_range, dist_mean)
    elif dist_per_bin is None and mean_per_bin is not None:
        dist_per_bin = np.interp(mean_per_bin, dist_mean[::-1],
                                 dist_range[::-1])

    # var_per_bin vs disp_per_bin
    if var_per_bin is None and disp_per_bin is not None:
        var_per_bin = mvr(mean_per_bin, disp_per_bin)
    elif disp_per_bin is None and var_per_bin is not None:
        disp_per_bin = inverse_mvr(mean_per_bin, var_per_bin)

    # pixel_var vs pixel_disp
    if pixel_var is None and pixel_disp is not None:
        pixel_var = mvr(pixel_mean, pixel_disp)
    elif pixel_disp is None and pixel_var is not None:
        pixel_disp = inverse_mvr(pixel_mean, pixel_var)
    else:
        raise ValueError('exactly one of pixel_var and pixel_disp must be '
                         'passed')

    # pixel_var_fit vs pixel_disp_fit
    if pixel_var_fit is None and pixel_disp_fit is not None:
        pixel_var_fit = mvr(pixel_mean, pixel_disp_fit)
    elif pixel_disp_fit is None and pixel_var_fit is not None:
        pixel_disp_fit = inverse_mvr(pixel_mean, pixel_var_fit)

    # convert pixel_* to dist_* quantities part 2: the fitted values
    dist_var_fit = None
    dist_disp_fit = None
    if dist_mean is not None and pixel_disp_fit is not None:
        dist_disp_fit = np.array([np.mean(pixel_disp_fit[pixel_dist == d])
                                  for d in dist_range])
        dist_var_fit = mvr(dist_mean, dist_disp_fit)

    # establish which y values will go into the cloud, bins, and fit
    if yaxis == 'var':
        cloud_y = pixel_var
        bin_y = var_per_bin
        fit_y = dist_var_fit if fit_align_dist else pixel_var_fit
        fit_label = r'$\hat{\sigma}^2$'
        ylabel = 'variance'
    elif yaxis == 'disp':
        cloud_y = pixel_disp
        bin_y = disp_per_bin
        fit_y = dist_disp_fit if fit_align_dist else pixel_disp_fit
        fit_label = r'$\hat{\alpha}$'
        ylabel = 'dispersion'
    else:
        raise ValueError('yaxis must be \'disp\' or \'var\'')

    # establish which x values will go into the cloud, bins, and fit
    if xaxis == 'dist':
        cloud_x = pixel_dist
        bin_x = dist_per_bin
        fit_x = dist_range if fit_align_dist else pixel_dist
        xlabel = 'distance'
    elif xaxis == 'mean':
        cloud_x = pixel_mean
        bin_x = mean_per_bin
        fit_x = dist_mean if fit_align_dist else pixel_mean
        xlabel = 'mean'
    else:
        raise ValueError('yaxis must be \'disp\' or \'var\'')

    # determine reasonable plot limits
    if xaxis == 'mean':
        xmin = mean_min
        xmax = np.percentile(pixel_mean, 99.99)
    elif xaxis == 'dist':
        xmin = 0
        xmax = dist_max
    else:
        raise ValueError('xaxis must be \'mean\' or \'dist\'')

    # prepare and apply filter indexes
    cloud_idx = np.isfinite(cloud_x) & np.isfinite(cloud_y) & \
        ((cloud_y > 0) if logy else True) & \
        ((cloud_x > 0) if logx else True) & \
        (cloud_x >= xmin) & (cloud_x <= xmax)
    cloud_x = cloud_x[cloud_idx]
    cloud_y = cloud_y[cloud_idx]
    try:
        bin_idx = np.isfinite(bin_x) & np.isfinite(bin_y) & \
            ((bin_y > 0) if logy else True) & \
            ((bin_x > 0) if logx else True) & \
            (bin_x >= xmin) & (bin_x <= xmax)
        bin_x = bin_x[bin_idx]
        bin_y = bin_y[bin_idx]
    except TypeError:
        pass
    fit_idx = np.isfinite(fit_x) & np.isfinite(fit_y) & \
        ((fit_y > 0) if logy else True) & \
        ((fit_x > 0) if logx else True) & \
        (fit_x >= xmin) & (fit_x <= xmax)
    fit_x = fit_x[fit_idx]
    fit_y = fit_y[fit_idx]

    if hexbin:
        ymin = max(np.percentile(cloud_y, 0.1), 1e-7) if logy \
            else min(np.percentile(cloud_y, 0.1), 0)
        ymax = np.percentile(cloud_y, 99.9)
    else:
        ymin = None
        ymax = None

    # override the default plot limits above if xlim/ylim were passed
    if xlim is not None:
        xmin, xmax = xlim
    if ylim is not None:
        ymin, ymax = ylim

    # hexbin per-pixel values
    if hexbin:
        scatter(cloud_x, cloud_y, logx=logx, logy=logy, hexbin=True,
                xlim=[xmin, xmax], ylim=[ymin, ymax])

    # scatter per-bin estimates
    if bin_x is not None and bin_y is not None:
        plt.scatter(bin_x, bin_y, label='%s per bin' % fit_label, color='C1',
                    s=scatter_size)

    # plot or scatter fitted estimates
    if fit_x is not None and fit_y is not None:
        sort_idx = np.argsort(fit_x) if not fit_align_dist \
            else np.arange(fit_x.shape[0])
        if scatter_fit > 0:
            plt.scatter(fit_x[sort_idx][::fit_x.shape[0]/scatter_fit],
                        fit_y[sort_idx][::fit_x.shape[0]/scatter_fit],
                        label=fit_label, color='C4', s=scatter_size)
        elif scatter_fit == -1:
            plt.plot(fit_x[sort_idx], fit_y[sort_idx],
                     label='smoothed %s' % fit_label, linewidth=3, color='C4')

    # add poisson line
    if xaxis == 'mean':
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
    else:
        if yaxis == 'disp' and not logy:
            plt.hlines(0, xmin, xmax, label='Poisson', linestyle='--',
                       linewidth=3, color='C3')

    # cleanup
    if hexbin:
        plt.ylim((ymin, ymax))
        plt.xlim((xmin, xmax))
    if not hexbin:
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()


@plotter
def plot_ddr(dist_per_bin, disp_per_bin, disp_fn, scatter_size=36, legend=True,
             **kwargs):
    """
    Simplified plotter to visualized distance-dispersion relationships.

    Plots per-distance dispersion estimates and the fitted dispersion curve.

    Parameters
    ----------
    dist_per_bin, disp_per_bin : np.ndarray
        The distances and estimated dispersions for each distance, respectively.
    disp_fn : function
        The smoothed dispersion function. Returns the smoothed dispersion as a
        function of distance.
    scatter_size : int
        The marker size when plotting scatter plots.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    xmin = dist_per_bin.min()
    xmax = dist_per_bin.max()
    xs = np.arange(xmin, xmax + 1)
    ys = disp_fn(xs)
    plt.scatter(dist_per_bin, disp_per_bin, label=r'$\hat{\alpha}$ per bin',
                color='C1', s=scatter_size)
    plt.plot(xs, ys, label=r'smoothed $\hat{\alpha}$', color='C4', lw=3)
    plt.hlines(0, xmin, xmax, label='Poisson', color='C3', lw=3, linestyle='--')
    plt.ylabel('dispersion')
    plt.xlabel('distance')
    if legend:
        plt.legend()
