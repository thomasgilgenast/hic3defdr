import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter


@plotter
def plot_fdr(eval_results, labels, colors=None, p_alt=None, **kwargs):
    """
    Plots an FDR control curve from a list of ``eval.npz``-style results.

    Parameters
    ----------
    eval_results : list of dict-like
        The dicts should have keys 'thresh' and 'fdr' whose values are parallel
        vectors describing the thresholds and FDRs to use for the FDR control
        curve. Each dict in the list represents a different FDR control curve
        which will be overlayed in the plot.
    labels : list of str
        List of labels parallel to ``eval_results`` providing names for each
        curve.
    colors : list of valid matplotlib colors, optional
        Colors for each FDR curve. Pass None to automatically assign colors.
    p_alt : float, optional
        Pass the true proportion of alternative (non-null) points to draw a
        dashed line representing the optimal BH-FDR control line and shade the
        zone of successful FDR control. Pass None to draw a dashed line
        indicating the boundary of successful FDR control.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # resolve colors
    if colors is None:
        colors = ['C%i' % i for i in range(len(labels))]

    for i, (res, label, color) in enumerate(zip(eval_results, labels, colors)):
        # unbox results
        fdr = res['fdr']
        thresh = res['thresh']

        # plot
        fdr_idx = np.isfinite(fdr)
        plt.plot(1 - thresh[fdr_idx], fdr[fdr_idx], color=color, label=label)

    # dashed line and shading (if requested)
    if p_alt is None:
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', zorder=1)
    else:
        plt.fill_between([0, 1], [0, 0], [0, 1], color='lightgray')
        plt.plot([0, 1], [0, 1-p_alt], color='gray', linestyle='--', zorder=1)

    # limits, labels, legend
    plt.axis('scaled')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FDR threshold')
    plt.ylabel('FDR')
    plt.legend(loc='upper left')
