import numpy as np
import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter

try:
    from sklearn.metrics import auc

    sklearn_avail = True
except ImportError:
    sklearn_avail = False
    auc = None


@plotter
def plot_roc(eval_results, labels, colors=None, **kwargs):
    # check for sklearn
    if not sklearn_avail:
        raise ImportError('failed to import scikit-learn - is it installed?')

    # resolve colors
    if colors is None:
        colors = ['C%i' % i for i in range(len(labels))]

    for i, (res, label, color) in enumerate(zip(eval_results, labels, colors)):
        # unbox results
        fpr = res['fpr']
        tpr = res['tpr']
        thresh = res['thresh']

        # plot
        plt.plot(fpr, tpr, color=color,
                 label='%s (AUC = %0.2f)' % (label, auc(fpr, tpr)))

        # annotate thresholds if there's only one curve
        if len(labels) == 1:
            thresh_idx = np.arange(len(thresh))[slice(1, None, len(thresh)/10)]
            for t in thresh_idx:
                plt.annotate(
                    '%.2f' % (1 - thresh[t]), xy=(fpr[t], tpr[t]),
                    xytext=(fpr[t] + 0.05, tpr[t] - 0.05),
                    arrowprops=dict(facecolor='k', width=3, headwidth=6,
                                    headlength=10))

    # 45-degree line, limits, labels, legend
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.axis('scaled')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
