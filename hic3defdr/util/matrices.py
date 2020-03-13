from functools import reduce

import numpy as np
import scipy.sparse as sparse
from scipy.ndimage import zoom


def deconvolute(matrix, bias, invert=False):
    """
    Applies bias factors to a sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        The matrix to bias. Will be converted to CSR by this function.
    bias : np.ndarray
        The dense bias vector.
    invert : bool
        Whether or not to invert the bias before applying it. By default this
        function multiplies the matrix by the bias. Infinite bias (1 / 0) will
        be converted to zero bias to allow rows with infinite bias to be dropped
        from the resulting sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The deconvoluted matrix.
    """
    csr = matrix.tocsr()
    if invert:
        inf_idx = bias == 0
        bias[inf_idx] = 1
        bias = 1 / bias
        bias[inf_idx] = 0
    bias_csr = sparse.diags([bias], [0])
    biased_csr = bias_csr.dot(csr)
    biased_csr = biased_csr.dot(bias_csr)
    return biased_csr


def wipe_distances(matrix, min_dist, max_dist):
    """
    Eliminates entries from a sparse matrix outside of a specified distance
    range.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        The matrix to wipe. Will be converted to COO by this function.
    min_dist, max_dist : int
        The minimum and maximum distance allowed, respectively, in bin units.

    Returns
    -------
    scipy.sparse.coo_matrix
        The wiped matrix.
    """
    coo = matrix.tocoo()
    dist = coo.col - coo.row
    coo.data[(dist < min_dist) | (dist > max_dist)] = 0
    coo.eliminate_zeros()
    return coo


def sparse_intersection(fnames, bias=None):
    """
    Computes the intersection set of (row, col) pairs across multiple sparse
    matrices.

    Parameters
    ----------
    fnames : list of str
        File paths to sparse matrices loadable by ``scipy.sparse.load_npz()``.
        Will be converted to COO by this function.
    bias : np.ndarray, optional
        Pass the bias matrix to drop rows and columns with bias factors of
        zero in any replicate.

    Returns
    -------
    row, col : np.ndarray
        The intersection set of (row, col) pairs.
    """
    csr_sum_coo = reduce(
        lambda x, y: x + y,
        ((deconvolute(sparse.load_npz(fname), bias[:, i]) > 0).astype(int)
         for i, fname in enumerate(fnames))).tocoo()
    full_idx = csr_sum_coo.data == len(fnames)
    return csr_sum_coo.row[full_idx], csr_sum_coo.col[full_idx]


def sparse_union(fnames, dist_thresh=1000, bias=None, size_factors=None,
                 mean_thresh=0.0):
    """
    Computes the intersection set of (row, col) pairs across multiple sparse
    matrices.

    Parameters
    ----------
    fnames : list of str
        File paths to sparse matrices loadable by ``scipy.sparse.load_npz()``.
        Will be converted to COO by this function.
    dist_thresh : int
        The maximum distance allowed, respectively, in bin units.
    bias : np.ndarray, optional
        Rectangular matrix containing bias factors for each bin (rows) and each
        replicate (columns).
    size_factors : np.ndarray, optional
        Size factors for each replicate.
    mean_thresh : float
        Minimum mean value (in normalized space) to keep pixels for.

    Returns
    -------
    row, col : np.ndarray
        The union set of (row, col) pairs.
    """
    if size_factors is None:
        size_factors = np.ones(len(fnames))
    csr_sum_coo = reduce(
        lambda x, y: x + y,
        (wipe_distances((deconvolute(sparse.load_npz(fname), bias[:, i],
                                     invert=True)
                         if bias is not None else sparse.load_npz(fname))
                        / size_factors[i], 0, dist_thresh)
         for i, fname in enumerate(fnames))).tocoo()
    full_idx = (csr_sum_coo.data >= len(fnames)*mean_thresh) \
        & np.isfinite(csr_sum_coo.data)
    return csr_sum_coo.row[full_idx], csr_sum_coo.col[full_idx]


def select_matrix(row_slice, col_slice, row, col, data, symmetrize=True):
    """
    Slices out a dense matrix from COO-formatted sparse data, filling empty
    values with ``np.nan``.

    Parameters
    ----------
    row_slice, col_slice : slice
        Row and column slice to use, respectively.
    row, col, data : np.ndarray
        COO-format sparse matrix to be sliced.
    symmetrize : bool
        Pass True to fill in lower-triangle points of the matrix.

    Returns
    -------
    np.ndarray
        Dense matrix.
    """
    r_start, r_stop = row_slice.start, row_slice.stop
    c_start, c_stop = col_slice.start, col_slice.stop
    idx = (row >= r_start) & (row < r_stop) & (col >= c_start) & (col < c_stop)
    matrix = np.ones((r_stop - r_start, c_stop - c_start)) * np.nan
    matrix[row[idx]-r_start, col[idx]-c_start] = data[idx]
    if symmetrize:
        t_idx = (col >= r_start) & (col < r_stop) & (row >= c_start) & \
            (row < c_stop)
        matrix[col[t_idx]-r_start, row[t_idx]-c_start] = data[t_idx]
    return matrix


def dilate(matrix, doublings):
    """
    Doubles the "resolution" of a matrix using nearest-neighbor interpolation.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    doublings : int
        The number of times the resolution will be doubled.

    Returns
    -------
    np.ndarray
        The dilated matrix.
    """
    for _ in range(doublings):
        matrix = zoom(matrix, 2, order=0)
    return matrix
