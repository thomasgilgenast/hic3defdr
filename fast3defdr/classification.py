import numpy as np
import scipy.sparse as sparse

from fast3defdr.clusters import find_clusters


def classify(row, col, value, clusters):
    """

    Parameters
    ----------
    row, col : np.ndarray
        Row and column indices, respectively, identifying pixels which
        correspond to rows of ``value``.
    value : np.ndarray
        Value to use to assign classes. Rows correspond to pixels, columns to
        conditions.
    clusters : list of set of tuple
        Only pixels in these clusters will be assigned classes.

    Returns
    -------
    list of list of set of tuple
        The outer list represents classes (one per condition). Its elements
        represent clusters of points with that class/condition.
    """
    # convert clusters to idx, parallel to row/col
    pixels = set().union(*(c for c in clusters))
    idx = np.zeros_like(row, dtype=bool)
    for i, p in enumerate(zip(row, col)):
        if p in pixels:
            idx[i] = True

    # assign classes
    classes = np.argmax(value[idx, :], axis=1)

    # re-cluster each class
    n = max(row.max(), col.max())
    class_clusters = []
    for c in range(value.shape[1]):
        coo = sparse.coo_matrix(
            (np.ones((classes == c).sum()),
             (row[idx][classes == c], col[idx][classes == c])), shape=(n, n))
        class_clusters.append(find_clusters(coo))

    return class_clusters
