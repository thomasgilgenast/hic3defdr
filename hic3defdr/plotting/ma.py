import numpy as np
import mpl_scatter_density  # noqa, side-effect on import
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D

from lib5c.util.plotting import plotter


@plotter
def plot_ma(data, sig_idx, loop_idx=None, names=None, s=-1, nonloop_s=None,
            density_dpi=72, vmax=None, nonloop_vmax=None, ax=None, legend=True,
            **kwargs):
    """
    Plots an MA plot.

    Parameters
    ----------
    data : np.ndarray
        Unlogged values to use for computing M and A for each pixel (rows) in
        each condition (columns).
    sig_idx : np.ndarray
        Boolean vector indicating which pixels among those that are in
        ``loop_idx`` are significantly differential.
    loop_idx : np.ndarraym, optional
        Boolean vector indicating which pixels in ``data`` are in loops. Pass
        None if all pixels in ``data`` are in loops.
    names : tuple of str, optional
        The names of the two conitions being compared.
    s : float
        The marker size to pass to `ax.scatter()`. Pass -1 to use
        `ax.scatter_density()` instead, avoiding plotting each point separately.
        See Notes below for more details and caveats.
    nonloop_s : float, optional
        Pass a separate marker size to use specifically for the non-loop pixels
        if `loop_idx` is not None. Useful for drawing just the non-loop pixels
        as a density by passing `s=1, nonloop_s=-1`. Pass None to use `s` as the
        size for both loop and non-loop pixels.
    density_dpi : int
        If `s` is -1 this specifies the DPI to use for the density grid.
    vmax, nonloop_vmax : float, optional
        The vmax to use for `ax.scatter_density()` if `s` or `nonloop_s` is -1,
        respectively. Pass None to choose values automatically.
    ax : pyplot axis
        The axis to plot to. Must have been created with
        `projection='scatter_density'`. Pass None to create a new axis.
    legend : bool
        Pass True to add a legend. Note that passing `legend='outside'` is not
        supported.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.

    Notes
    -----
    It is recommended to use `ax.scatter_density()` from the
    `mpl_scatter_density` module to plot the data by passing `s=-1`. This avoids
    the massive slowdown experienced when plotting one separate dot for each
    pixel in the genome in the case where `loop_idx=None`. In order to support
    `ax.scatter_density()`, a new pyplot figure and axis must be created with
    `projection='scatter_density'`. Therefore, even though this function is
    decorated with `@plotter`, it has a non-standard behavior for interpreting
    the `ax` kwarg: if `ax=None` is passed, it will create a new figure and axis
    rather than re-using the current axis.
    """
    # compute MA
    m = np.log2(data[:, 0]) - np.log2(data[:, 1])
    a = 0.5 * (np.log2(data[:, 0]) + np.log2(data[:, 1]))

    # resolve ax
    if ax is None:
        if s == -1 or nonloop_s == -1:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        else:
            ax = plt.gca()
    else:
        if (s == -1 or nonloop_s == -1) and not hasattr(ax, 'scatter_density'):
            raise ValueError('ax passed to plot_ma() was not created with '
                             '`projection=\'scatter_density\'`')

    # resolve vmax and create normalization objects
    vmax = loop_idx.sum()/1000. if not vmax else vmax
    nonloop_vmax = (len(loop_idx)-loop_idx.sum())/1000. \
        if not nonloop_vmax else nonloop_vmax
    norm = SymLogNorm(1., vmin=0, vmax=vmax)
    nonloop_norm = SymLogNorm(1., vmin=0, vmax=nonloop_vmax)

    # resolve scatter_fn and scatter_kwargs
    scatter_fn = ax.scatter_density if s == -1 else ax.scatter
    scatter_kwargs = {'dpi': density_dpi, 'norm': norm} if s == -1 else {'s': s}
    nonloop_fn = scatter_fn if nonloop_s is None \
        else ax.scatter_density if nonloop_s == -1 else ax.scatter
    nonloop_kwargs = dict(scatter_kwargs) if nonloop_s is None \
        else {'dpi': density_dpi, 'norm': nonloop_norm} if nonloop_s == -1 \
        else {'s': s}
    if nonloop_s is None and s == -1:
        nonloop_kwargs['norm'] = nonloop_norm

    # apparatus for tracking plot limits
    # we need to do this because ax.scatter_density() resets the axes limits on
    # each call; if s != -1 we will leave the limits alone
    limits = {'xmin': 0, 'xmax': 0, 'ymax': 0}

    def update_lim():
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        limits['xmin'] = min(xmin, limits['xmin'])
        limits['xmax'] = max(xmax, limits['xmax'])
        limits['ymax'] = max(ymax, -ymin, limits['ymax'])

    # prepare colors and labels
    colors = ['gray', 'purple', 'red', 'blue']
    labels = [
        'non-loop pixels',
        'constitutive loop pixels',
        '%s-specific loop pixels' % names[0]
        if names is not None else 'weakening loop pixels',
        '%s-specific loop pixels' % names[1]
        if names is not None else 'strengthening loop pixels'
    ]

    # plot
    include_non_loops = False
    if loop_idx is None:
        loop_idx = np.ones_like(a, dtype=bool)
    else:
        include_non_loops = True
        # non-loop pixels
        nonloop_fn(a[~loop_idx], m[~loop_idx], color=colors[0], label=labels[0],
                   **nonloop_kwargs)
        update_lim()
    # constitutive pixels
    scatter_fn(a[loop_idx][~sig_idx], m[loop_idx][~sig_idx],
               color=colors[1], label=labels[1], **scatter_kwargs)
    update_lim()
    # differential pixels
    scatter_fn(a[loop_idx][sig_idx][m[loop_idx][sig_idx] > 0],
               m[loop_idx][sig_idx][m[loop_idx][sig_idx] > 0],
               color=colors[2], label=labels[2], **scatter_kwargs)
    update_lim()
    scatter_fn(a[loop_idx][sig_idx][m[loop_idx][sig_idx] < 0],
               m[loop_idx][sig_idx][m[loop_idx][sig_idx] < 0],
               color=colors[3], label=labels[3], **scatter_kwargs)
    update_lim()

    # cleanup
    if s == -1 or nonloop_s == -1:
        ax.set_xlim((limits['xmin'], limits['xmax']))
        ax.set_ylim((-limits['ymax'], limits['ymax']))
    ax.set_xlabel('mean log interaction strength')
    ax.set_ylabel('log fold change' +
                  (' (%s over %s)' % tuple(names) if names is not None else ''))

    # add custom legend
    if legend:
        legend_elements = [
            Line2D(
                [0], [0], markersize=10, marker='o', markerfacecolor=color,
                label=label, color=(0, 0, 0, 0), markeredgecolor=(0, 0, 0, 0)
            )
            for color, label in zip(colors, labels)
        ]
        if not include_non_loops:
            legend_elements = legend_elements[1:]
        ax.legend(handles=legend_elements)
