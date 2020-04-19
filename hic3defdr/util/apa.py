import numpy as np

from hic3defdr.util.clusters import cluster_to_slices


def make_apa_stack(matrix, clusters, width, min_dist=None):
    """
    Creates a stack of square matrix slices centered on each cluster in a list.

    Parameters
    ----------
    matrix : scipy.spmatrix
        The matrix to take slices of.
    clusters : list of clusters
        The clusters around which to take the slices. Each cluster should be a
        list of [x, y] pairs representing the points in that cluster.
    width : int
        The width of the slice to take in bin units. Should be odd.
    min_dist : int, optional
        Clusters with interaction distances shorter than this (in bin units)
        will be given an slice of all nan's. Pass None to use a sane default.

    Returns
    -------
    np.ndarray
        This array will have three dimensions. The first axis represents the
        clusters, the next two represent the square slice for each cluster. The
        array may contain nan's.
    """
    matrix = matrix.tocsr()
    if min_dist is None:
        min_dist = width + 1
    stack = np.zeros((len(clusters), width, width))
    size = max(matrix.shape)
    r = int(width / 2)
    for idx, cluster in enumerate(clusters):
        com = np.mean([np.array(p) for p in cluster], axis=0)
        if np.abs(np.diff(com)) < min_dist or com[0] < r or com[1] < r or \
                size - com[0] < r or size - com[1] < r:
            stack[idx, :, :] = np.nan
        else:
            stack[idx, :, :] = \
                matrix[cluster_to_slices(cluster, width)].toarray()
    return stack
