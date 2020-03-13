import numpy as np
import scipy.sparse as sparse

from hic3defdr.util.banded_matrix import BandedMatrix


def filter_sparse_rows_count(matrix, min_nnz=25, k=300):
    """
    Wipes the sparse bins (both rows and columns) of a matrix, where "sparse"
    means that the bin has less than ``min_nnz`` nonzero interactions with both
    the ``k`` nearest upstream and downstream bins.

    This function wipes with zeros and will eliminate the wiped positions if
    ``matrix`` is a CSR format sparse matrix.

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csr_matrix, or BandedMatrix
        The matrix to wipe sparse bins from.
    min_nnz : int
        The minimum number of nonzero contacts a bin has to make in either
        direction (upstream or downstream) to escape being wiped.
    k : int
        How many bins upstream or downstream to look when counting the number
        of nonzero contacts a bin makes in a given direction.

    Returns
    -------
    np.ndarray, scipy.sparse.csr_matrix, or BandedMatrix
        The filtered matrix.
    """
    matrix = matrix.copy()
    if min_nnz == 0 or k == 0:
        return matrix
    if isinstance(matrix, np.ndarray):
        # matrix already supports row/col indexing
        # BandedMatrix can be created directly from matrix
        bm = BandedMatrix.from_ndarray(matrix, max_range=k)
    else:
        if BandedMatrix.is_bandedmatrix(matrix):
            bm = matrix
        elif isinstance(matrix, sparse.spmatrix):
            if not isinstance(matrix, sparse.csr_matrix):
                # incoming matrix doesn't support row/col indexing, convert it
                matrix = matrix.tocsr()
            # create BandedMatrix directly from sparse matrix
            bm = BandedMatrix(matrix, max_range=k)
        else:
            raise ValueError('invalid input type')
    bm.symmetrize()
    us_rows = np.where((bm.offsets >= -k) & (bm.offsets < 0))[0]
    ds_rows = np.where((bm.offsets <= k) & (bm.offsets > 0))[0]
    deleted_indices = (np.sum(bm.data[us_rows, :] > 0, axis=0) < min_nnz) & \
                      (np.sum(bm.data[ds_rows, :] > 0, axis=0) < min_nnz)
    if isinstance(matrix, sparse.csr_matrix):
        keep_indices_csr = sparse.diags([~deleted_indices], [0], dtype=int)
        matrix = keep_indices_csr.dot(matrix)
        matrix = matrix.dot(keep_indices_csr)
    else:
        deleted_indices = np.where(deleted_indices)[0]
        matrix[:, deleted_indices] = 0
        matrix[deleted_indices, :] = 0
    return matrix
