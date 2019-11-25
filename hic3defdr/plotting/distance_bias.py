import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib5c.util.plotting import plotter


@plotter
def plot_distance_bias(ob, bins, bin_labels=None, idx='disp', threshold=0.05,
                       colors=None, labels=None, xlabel='distance range',
                       legend_label='group', **kwargs):
    """
    Plots a bar plot illustrating the degree to which p-values are biased among
    different distance scales.

    This method visualizes distance bias by computing a specified percentile of
    all p-values called in an analysis, and then computing the proportion of
    pixels with p-values below this percentile in each distance bin.

    Parameters
    ----------
    ob : hic3defdr.HiC3DeFDR object or list of hic3defdr.HiC3DeFDR objects
        The analyis or analyses to inspect for distance bias.
    bins : list of tuple of int
        Each tuple represents a distance bin as a (min, max) pair, where min and
        max are distances in bin units and the ranges are inclusive. If either
        min or max is None, the distance bin will be considered unbounded on
        that end.
    bin_labels : list of str
        Pass a list of labels to describe the distance bins.
    idx : {'disp', 'loop'}
        Pass 'disp' to use p-values for all points for which dispersion was
        estimated. Pass 'loop' to only use p-values for points which are in
        loops.
    threshold : float
        The percentile to use for the comparison.
    colors : str or list of str, optional
        If ``ob`` is a single object, pass a single color to color the bars in
        the barplot. If ``ob`` is a list of objects, pass a list of colors. Pass
        None to use automatic colors.
    labels : list of str, optional
        If ``ob`` is a list of objects, you must pass a list of strings to label
        the objects. Otherwise, this kwarg does nothing.
    xlabel : str
        The label for the x-axis.
    legend_label : str
        If ``ob`` is a list of objects, the label to use for the legend title.
        Otherwise, this kwarg does nothing.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis
        The axis plotted on.
    """
    # promote ob to list and resolve labels
    if type(ob) not in [list, tuple]:
        ob = [ob]
        hue = None
        color = 'k' if colors is None else colors
        colors = None
        if labels is None:
            labels = ['group1']
    else:
        hue = legend_label
        color = None
        if labels is None:
            raise ValueError('must pass labels if ob is a list or tuple')

    # resolve bin_labels
    if bin_labels is None:
        bin_labels = []
        for min_dist, max_dist in bins:
            if min_dist is None and max_dist is not None:
                label = '<= %s' % max_dist
            elif min_dist is not None and max_dist is None:
                label = '>= %s' % min_dist
            elif min_dist is None and max_dist is None:
                label = 'all'
            else:
                label = '%s to %s' % (min_dist, max_dist)
            bin_labels.append(label)

    data = []
    for o, label in zip(ob, labels):
        # load data
        disp_idx, _ = o.load_data('disp_idx', 'all')
        if idx == 'loop':
            loop_idx, _ = o.load_data('loop_idx', 'all')
            rc_idx = (disp_idx, loop_idx)
            p_idx = loop_idx
        else:
            rc_idx = disp_idx
            p_idx = None
        disp_idx, _ = o.load_data('disp_idx', 'all')
        loop_idx, _ = o.load_data('loop_idx', 'all')
        row, _ = o.load_data('row', 'all', idx=rc_idx)
        col, _ = o.load_data('col', 'all', idx=rc_idx)
        dist = col - row
        pvalues, _ = o.load_data('pvalues', 'all', idx=p_idx)
        p_star = np.percentile(pvalues, 100*threshold)

        # process each distance bin
        for bin_label, (min_dist, max_dist) in zip(bin_labels, bins):
            dist_idx = np.ones(len(dist), dtype=bool)
            if min_dist is not None:
                dist_idx[dist < min_dist] = False
            if max_dist is not None:
                dist_idx[dist > max_dist] = False
            perc = np.mean(pvalues[dist_idx] < p_star)
            data.append({legend_label: label, xlabel: bin_label,
                         'percentage significant': perc})
    df = pd.DataFrame(data)
    sns.barplot(data=df, x=xlabel, y='percentage significant', hue=hue,
                color=color, palette=colors)
    xlim = plt.xlim()
    plt.hlines(threshold, *xlim, color='gray', linestyle='--')
    plt.xlim(xlim)
