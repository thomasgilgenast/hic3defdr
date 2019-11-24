import numpy as np
import scipy.sparse as sparse

from hic3defdr.util.clusters import find_clusters


def threshold_and_cluster(qvalues, row, col, fdr):
    """
    Thresholds pixels by comparing their q-values to a target FDR and clusters
    the significant and insignificant pixels.

    Parameters
    ----------
    qvalues : np.ndarray
        The qvalue for each pixel under consideration.
    row, col : np.ndarray
        The row and column indices corresponding to the qvalues.
    fdr : float
        The FDR to threshold on.

    Returns
    -------
    sig_clusters, insig_clusters : list of set of tuple of int
        Lists of the significant and insignificant clusters, respectively.
    """
    # threshold on FDR
    sig_idx = qvalues < fdr
    insig_idx = qvalues >= fdr

    # gather and cluster sig and insig points
    n = max(row.max(), col.max()) + 1  # guess matrix shape
    sig_points = sparse.coo_matrix(
        (np.ones(sig_idx.sum(), dtype=bool),
         (row[sig_idx],
          col[sig_idx])), shape=(n, n))
    insig_points = sparse.coo_matrix(
        (np.ones(insig_idx.sum(), dtype=bool),
         (row[insig_idx],
          col[insig_idx])), shape=(n, n))
    sig_clusters = find_clusters(sig_points)
    insig_clusters = find_clusters(insig_points)
    return sig_clusters, insig_clusters


def size_filter(clusters, cluster_size):
    """
    Filters out clusters which are smaller than cluster_size.

    Parameters
    ----------
    clusters : list of set of tuple of int
        The clusters to filter.
    cluster_size : int
        The minimum size of a cluster needed to pass this filter.

    Returns
    -------
    list of set of tuple of int
        The filtered clusters.
    """
    return [c for c in clusters if len(c) >= cluster_size]
