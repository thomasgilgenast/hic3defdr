import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter


@plotter
def plot_dd_curves(row, col, before, after, design, log=True, **kwargs):
    """
    Plots a comparison of distance dependence curves before and after scaling.

    Parameters
    ----------
    row, col : np.ndarray
        Row and column indices identifying the location of pixels in ``before``
        and ``after``.
    before, after: np.ndarray
        Counts per pixel before and after scaling, respectively.
    design : pd.DataFrame
        DataFrame with boolean dtype whose rows correspond to replicates and
        whose columns correspond to conditions. Replicate and condition names
        will be inferred from the row and column labels, respectively.
    log : bool
        Pass True to log both axes of the distance dependence plot.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    dist = col - row
    n = max(row.max(), col.max()) + 1
    dist_bin_idx = np.digitize(dist, np.linspace(0, 1000, 101), right=True)
    bs = np.arange(1, dist_bin_idx.max() + 1)
    before_means = np.array(
        [np.sum(before[dist_bin_idx == b, :], axis=0) / float(n - b)
         for b in bs])
    after_means = np.array(
        [np.sum(after[dist_bin_idx == b, :], axis=0) / float(n - b)
         for b in bs])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for r, repname in enumerate(design.index):
        ax1.plot(bs, before_means[:, r], label=repname, color='C%i' % r)
        ax2.plot(bs, after_means[:, r], label=repname, color='C%i' % r)
    plt.legend()
    ax1.set_xlabel('distance (bins)')
    ax2.set_xlabel('distance (bins)')
    ax1.set_ylabel('average counts')
    ax1.set_title('before scaling')
    ax2.set_title('after scaling')
    if log:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
