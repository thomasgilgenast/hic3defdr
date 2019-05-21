import numpy as np
import scipy.sparse as sparse
from scipy.ndimage import zoom


def sparse_intersection(fnames):
    """
    Computes the intersection set of (row, col) pairs across multiple sparse
    matrices.

    Parameters
    ----------
    fnames : list of str
        File paths to sparse matrices loadable by ``scipy.sparse.load_npz()``.
        Will be converted to COO by this function.

    Returns
    -------
    row, col : np.ndarray
        The intersection set of (row, col) pairs.
    """
    csr_sum_coo = reduce(
        lambda x, y: x + y, ((sparse.load_npz(fname) > 0).astype(int)
                             for fname in fnames)).tocoo()
    full_idx = csr_sum_coo.data == len(fnames)
    return csr_sum_coo.row[full_idx], csr_sum_coo.col[full_idx]


def select_matrix(row_slice, col_slice, row, col, data):
    """
    Slices out a dense matrix from COO-formatted sparse data, filling empty
    values with ``np.nan``.

    Parameters
    ----------
    row_slice, col_slice : slice
        Row and column slice to use, respectively.
    row, col, data : np.ndarray
        COO-format sparse matrix to be sliced.

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
