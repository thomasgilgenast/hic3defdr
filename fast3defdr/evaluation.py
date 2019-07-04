import numpy as np

from fast3defdr.progress import tqdm_maybe as tqdm

try:
    from sklearn.metrics import roc_curve, confusion_matrix

    sklearn_avail = True
except ImportError:
    sklearn_avail = False
    roc_curve = None
    confusion_matrix = None


def make_y_true(row, col, clusters, labels):
    """
    Makes a boolean vector of the true labels for each pixel, given a list of
    clusters and the true label for each cluster.

    Parameters
    ----------
    row, col : np.ndarray
        The row and column indices of pixels to be labeled.
    clusters : list of list of tuple
        The outer list is a list of clusters. Each cluster is a list of (i, j)
        tuples marking the position of significant points which belong to that
        cluster.
    labels : list of str
        List of labels for each cluster, parallel to ``clusters``.

    Returns
    -------
    np.ndarray
        Boolean vector with the same length as ``row``/``col``. It's `i`th
        element is False when the pixel at `(row[i], col[i])` is in a cluster
        with label 'constit' and is True otherwise.
    """
    sig_idx = ~(labels == 'constit')
    sig_pixels = set().union(*[c for i, c in enumerate(clusters) if sig_idx[i]])
    return np.array([True if (r, c) in sig_pixels else False
                     for r, c in zip(row, col)])


def evaluate(y_true, qvalues):
    """
    Evaluates how good a vector of q-values (or p-values) is at predicting the
    vector of true labels.

    Parameters
    ----------
    y_true : np.ndarray
        The boolean vector of true labels.
    qvalues : np.ndarray
        Vector of q-values or p-values which are supposed to predict the boolean
        label in ``y_true``.

    Returns
    -------
    fdr, fpr, tpr, thresh : np.ndarray
        Parallel arrays of the FDR, FPR, TPR, and thresholds (in ``1 - qvalue``)
        space which specify the FDR, FPR, and and TPR at each threshold. The
        thresholds are selected to represent the convex edge of the ROC curve.
        The FDR will only be evaluated at about 100 selected thresholds and
        will be set to ``np.nan`` at the un-evaluated thresholds.
    """
    if not sklearn_avail:
        raise ImportError('failed to import scikit-learn - is it installed?')
    y_pred = 1 - qvalues
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    fdr = np.ones_like(fpr) * np.nan
    for i in tqdm(range(1, len(thresh), len(thresh)/100)):
        fdr[i] = compute_fdr(y_true, y_pred > thresh[i])
    return fdr, fpr, tpr, thresh


def compute_fdr(y_true, y_pred):
    """
    Computes the observed false discovery rate from boolean vectors of true and
    predicted labels.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Boolean vectors of the true and predicted labels, respectively.

    Returns
    -------
    float
        The false discovery rate.
    """
    if not sklearn_avail:
        raise ImportError('failed to import scikit-learn - is it installed?')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / float(fp + tp)
