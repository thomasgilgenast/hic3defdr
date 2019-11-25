import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib5c.util.plotting import plotter


@plotter
def plot_fn_vs_fp(eval_results, labels, threshold=0.15, colors=None,
                  xlabel='label', **kwargs):
    """
    Plots two bar plots, one showing FNR and the other showing FPR at a fixed
    threshold.

    Parameters
    ----------
    eval_results : list of dict-like or list of list of dict-like
        The dicts should have keys 'thresh', 'fpr', and 'tpr' whose values are
        parallel vectors describing the thresholds, FPRs, and TPRs to use for
        the bar plots. Each dict in the list represents a different bar which
        will be added to each bar plot. Pass a nested list of dicts to draw
        grouped bar plots, denoting the outer list by color and the inner list
        by x-axis position.
    labels : list of str or (list of str, list of str)
        List of labels parallel to ``eval_results`` providing names for each
        bar. If ``eval_results`` is a nested list, pass a tuple of two lists.
        The first list should provide the labels for the outer list grouping
        (``len(labels[0]) == len(eval_results)``) and the second should provide
        the labels for the inner list grouping
        (``len(labels[1]) == len(eval_results[i])`` for any ``i``).
    threshold : float
        The fixed threshold at which the FNR and FPR will be plotted. In
        practice, this function will use the closest threshold found in each
        dict in ``eval_results``.
    colors : matplotlib color or list of colors, optional
        Specify the color to use for the bars in the bar plot. If
        ``eval_results`` is a nested list, pass a list of colors to color core
        the outer list grouping (``len(colors) == len(eval_results)``). Pass
        None to use automatic colors.
    xlabel : str
        The label to use for the x-axis.
    kwargs : kwargs
        Typical plotter kwargs.

    Returns
    -------
    pyplot axis, array of pyplot axes
        The first pyplot axis returned is injected by ``@plotter``. The array of
        pyplot axes is the second return value from the call to
        ``plt.subplots()`` that is used to create the pair of barplots.
    """
    data = []
    if type(eval_results[0]) in [list, tuple]:
        hue = 'group'
        color = None
        palette = colors if colors else None
        for res_group, group_label in zip(eval_results, labels[0]):
            for res, label in zip(res_group, labels[1]):
                if res is None:
                    continue
                # unbox results
                fpr = res['fpr']
                fnr = 1 - res['tpr']
                thresh = 1 - res['thresh']

                # find closest thresh
                idx = np.argmin(np.abs(thresh - threshold))

                # append to data
                data.append({xlabel: label, 'group': group_label,
                             'FPR': fpr[idx], 'FNR': fnr[idx]})
    else:
        hue = None
        color = colors if colors else 'k'
        palette = None
        for res, label in zip(eval_results, labels):
            # unbox results
            fpr = res['fpr']
            fnr = 1 - res['tpr']
            thresh = 1 - res['thresh']

            # find closest thresh
            idx = np.argmin(np.abs(thresh - threshold))

            # append to data
            data.append({xlabel: label, 'FPR': fpr[idx], 'FNR': fnr[idx]})
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplots_adjust(wspace=0.4)
    sns.barplot(data=df, x=xlabel, y='FNR', hue=hue, palette=palette,
                color=color, ax=axes[0])
    sns.barplot(data=df, x=xlabel, y='FPR', hue=hue, palette=palette,
                color=color, ax=axes[1])
    if hue is not None:
        axes[0].legend_.remove()
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return axes
