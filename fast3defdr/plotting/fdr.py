import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter


@plotter
def plot_fdr(eval_results, labels, colors=None, **kwargs):
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

    # 45-degree line, limits, labels, legend
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.axis('scaled')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FDR threshold')
    plt.ylabel('FDR')
    plt.legend(loc='upper left')
