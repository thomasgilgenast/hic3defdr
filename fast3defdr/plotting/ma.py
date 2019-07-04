import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter


@plotter
def plot_ma(data, sig_idx, loop_idx=None, names=None, s=1, **kwargs):
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
        The marker size to use for the scatterplot.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # compute MA
    m = np.log2(data[:, 0]) - np.log2(data[:, 1])
    a = 0.5 * (np.log2(data[:, 0]) + np.log2(data[:, 1]))

    # plot
    if loop_idx is None:
        loop_idx = np.ones_like(a, dtype=bool)
    else:
        plt.scatter(a[~loop_idx], m[~loop_idx], s=s, color='lightgray',
                    label='non-loop pixels')
    plt.scatter(a[loop_idx][~sig_idx], m[loop_idx][~sig_idx], s=s,
                color='black', label='constitutive loop pixels')
    plt.scatter(a[loop_idx][sig_idx][m[loop_idx][sig_idx] > 0],
                m[loop_idx][sig_idx][m[loop_idx][sig_idx] > 0], s=s,
                color='red', label='%s-specific loop pixels' % names[0]
                if names is not None else 'weakening loop pixels')
    plt.scatter(a[loop_idx][sig_idx][m[loop_idx][sig_idx] < 0],
                m[loop_idx][sig_idx][m[loop_idx][sig_idx] < 0], s=s,
                color='blue', label='%s-specific loop pixels' % names[1]
                if names is not None else 'strengthening loop pixels')

    # cleanup
    plt.xlabel('mean log expression')
    plt.ylabel('log fold change' +
               (' (%s over %s)' % tuple(names) if names is not None else ''))
