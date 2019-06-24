import numpy as np

try:
    from sklearn.metrics import roc_curve, confusion_matrix

    sklearn_avail = True
except ImportError:
    sklearn_avail = False
    roc_curve = None
    confusion_matrix = None


def make_y_true(row, col, clusters, labels):
    sig_idx = ~(labels == 'constit')
    sig_pixels = set().union(*[c for i, c in enumerate(clusters) if sig_idx[i]])
    return np.array([True if (r, c) in sig_pixels else False
                     for r, c in zip(row, col)])


def evaluate(y_true, qvalues):
    """

    Parameters
    ----------
    y_true
    qvalues

    Returns
    -------
    fdr, fpr, tpr, thresh : np.ndarray
    """
    if not sklearn_avail:
        raise ImportError('failed to import scikit-learn - is it installed?')
    y_pred = 1 - qvalues
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    fdr = np.array([compute_fdr(y_true, y_pred > t) for t in thresh])
    return fdr, fpr, tpr, thresh


def compute_fdr(y_true, y_pred):
    if not sklearn_avail:
        raise ImportError('failed to import scikit-learn - is it installed?')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / float(fp + tp)
