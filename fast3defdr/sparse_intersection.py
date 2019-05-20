import numpy as np
import scipy.sparse as sparse


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


def sparse_intersection_slow(fnames):
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
    def fname_to_set(fname):
        coo = sparse.load_npz(fname).tocoo()
        return set(zip(coo.row, coo.col))
    s = reduce(lambda x, y: x.intersection(fname_to_set(y)), fnames[1:],
               fname_to_set(fnames[0]))
    row, col = zip(*s)
    return np.array(row, dtype=int), np.array(col, dtype=int)
