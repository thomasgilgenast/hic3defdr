import numpy as np
import scipy.sparse as sparse

from lib5c.util.system import check_outdir


def roll_footprint(footprint):
    """
    "Rolls" each row to the right by its row index, then reverses the row order
    and transposes the result. This is equivalent to the operation
    ``f(x) = sparse.dia_matrix(x).data`` where ``x`` is a dense matrix, as long
    as ``sparse.dia_matrix(x).offsets`` is a sequential matrix (this is
    guaranteed by the BandedMatrix subclass).

    The result of this is that the rolled footprint is suitable for use in
    convolution against ``sparse.dia_matrix.data``, for example as
    ``convolve(sparse.dia_matrix(x).data, roll_footprint(footprint))`` where
    ``x`` is a dense matrix and ``footprint`` is expressed in terms of the
    dense spatial coordinates.

    Parameters
    ----------
    footprint : np.ndarray
        The footprint to roll.

    Returns
    -------
    np.ndarray
        The rolled footprint.
    """
    n = len(footprint)
    m = 2*n - 1
    row_idx, col_idx = np.ogrid[:n, :m]
    col_idx = (col_idx + np.arange(n)[:, np.newaxis]) % m
    return np.hstack(
        [np.zeros((n, n-1), dtype=int), footprint])[row_idx, col_idx][::-1, :].T


class BandedMatrix(sparse.dia_matrix):
    def __init__(self, data, shape=None, copy=False, max_range=300,
                 upper=None):
        if shape is None:
            if isinstance(data, BandedMatrix):
                instance = data
            elif isinstance(data, sparse.dia_matrix):
                if not self.is_bandedmatrix(data):
                    raise ValueError('dia_matrix detected, but not a '
                                     'BandedMatrix')
                instance = self.from_dia_matrix(data, copy=copy)
            elif isinstance(data, sparse.spmatrix):
                instance = self.from_sparse(data, max_range=max_range,
                                            upper=upper)
            elif isinstance(data, np.ndarray):
                instance = self.from_ndarray(data, max_range=max_range,
                                             upper=upper)
            else:
                raise ValueError('invalid arguments passed to BandedMatrix '
                                 'constructor')
            self.__init__((instance.data, instance.offsets),
                          shape=instance.shape, copy=False)
        else:
            super(BandedMatrix, self).__init__(
                data, shape=shape, copy=copy)

    @classmethod
    def from_dia_matrix(cls, dia_matrix, copy=True):
        if copy:
            return cls((dia_matrix.data.copy(), dia_matrix.offsets.copy()),
                       shape=dia_matrix.shape)
        else:
            return cls((dia_matrix.data, dia_matrix.offsets),
                       shape=dia_matrix.shape)

    @classmethod
    def from_ndarray(cls, ndarray, max_range=300, upper=None):
        if upper is None:
            upper = np.all(np.tril(ndarray, k=-1) == 0)
        max_range = min(ndarray.shape[0], max_range)
        data = np.pad(
            cls.roll_matrix(ndarray.astype(float).T, max_range=max_range),
            ((1, 1), (0, 0)), mode='constant', constant_values=np.nan)
        offsets = np.arange(-max_range-1, max_range+2)
        instance = cls((data, offsets), shape=ndarray.shape)
        if upper:
            instance.make_upper()
        return instance

    @classmethod
    def from_sparse(cls, spmatrix, max_range=300, upper=None):
        """
        Constructs a BandedMatrix from a scipy.spare.spmatrix instance.

        Parameters
        ----------
        spmatrix : scipy.spare.spmatrix
            The sparse matrix to convert to BandedMatrix.
        max_range : int
            Offdiagnals beyond the max_range'th offdiagonal will be discarded.
        upper : bool, optional
            Pass True to keep only the upper triangular entries. Pass False to
            keep all the entries. Pass None to guess what to do based on whether
            or not the input matrix is upper triangular.

        Returns
        -------
        BandedMatrix
            The new BandedMatrix.

        Notes
        -----
        Mostly stolen from sparse.coo_matrix.todia(), plus nan padding from
        BandedMatrix.from_ndarray() and nan triangles from
        BandedMatrix.roll_matrix().
        """
        # guess upper if None was passed
        if upper is None:
            upper = sparse.tril(spmatrix, k=-1).nnz == 0

        # check empty bins to be wiped later
        idx = np.where(
            np.squeeze(np.asarray(spmatrix.tocsr().getnnz(0) == 0)))[0]

        # logic for converting COO to DIA-style data
        n = spmatrix.shape[0]
        max_range = min(n, max_range)
        coo = spmatrix.tocoo()
        coo.sum_duplicates()
        ks = coo.col - coo.row
        keep_idx = np.abs(ks) <= max_range
        ks = ks[keep_idx]
        js = coo.col[keep_idx]
        vs = coo.data[keep_idx]
        data = np.zeros((max_range*2 + 1, n))
        data[ks + max_range, js] = vs

        # add padding and nan triangles to data
        data[np.tril(np.ones_like(data, dtype=bool), k=-max_range-1)] = np.nan
        data[np.triu(np.ones_like(data, dtype=bool), k=n-max_range)] = np.nan
        data = np.pad(data, ((1, 1), (0, 0)), mode='constant',
                      constant_values=np.nan)

        # construct BandedMatrix instance
        instance = cls((data, np.arange(-max_range - 1, max_range + 2)),
                       shape=spmatrix.shape)

        # wipe empty bins with nan
        instance[idx, :] = np.nan
        instance[:, idx] = np.nan

        # honor upper
        if upper:
            instance.make_upper()

        return instance

    @classmethod
    def is_bandedmatrix(cls, x):
        if type(x) == str:
            try:
                instance = sparse.load_npz(x)
            except (IOError, ValueError):
                return False
        else:
            instance = x
        if not isinstance(instance, sparse.dia_matrix):
            return False
        if not np.all(np.diff(instance.offsets) == 1):
            return False
        if not np.all(np.isnan(instance.data[0, :])):
            return False
        if not np.all(np.isnan(instance.data[-1, :])):
            return False
        return True

    def is_upper(self):
        return np.all(self.offsets >= -1)

    def max_range(self):
        return np.max(np.abs(self.offsets)) - 1

    @classmethod
    def load(cls, fname, **kwargs):
        if fname.endswith('.npy'):
            return cls(np.load(fname), **kwargs)
        return cls(sparse.load_npz(fname), **kwargs)

    def save(self, fname):
        check_outdir(fname)
        sparse.save_npz(fname, self)

    def copy(self):
        return self.from_dia_matrix(self)

    @classmethod
    def align(cls, *matrices):
        common_offsets = set.intersection(*(set(m.offsets) for m in matrices))
        res = []
        for m in matrices:
            extra_offsets = set(m.offsets) - common_offsets
            if extra_offsets:
                extra_offsets = np.array(list(extra_offsets))
                res.append(cls(
                    (np.delete(m.data, extra_offsets, axis=0),
                     np.delete(m.offsets, extra_offsets)), shape=m.shape))
            else:
                res.append(m)
        return res

    @staticmethod
    def roll_matrix(matrix, max_range=300):
        n = len(matrix)
        m = 2*max_range + 1
        row_idx, col_idx = np.ogrid[:n, :m]
        col_idx = (col_idx + np.arange(n)[:, np.newaxis] - max_range) % n
        rolled_matrix = matrix[row_idx, col_idx].T[::-1, :]
        rolled_matrix[np.tril(np.ones_like(rolled_matrix, dtype=bool),
                              k=-max_range-1)] = np.nan
        rolled_matrix[np.triu(np.ones_like(rolled_matrix, dtype=bool),
                              k=n-max_range)] = np.nan
        return rolled_matrix

    @staticmethod
    def make_mask(matrix, min_range=None, max_range=None, upper=False,
                  nan=False):
        """
        General-purpose function for creating masks for contact matrices.

        Parameters
        ----------
        matrix : np.ndarray
            Square contact matrix to make a mask for.
        min_range, max_range : int, optional
            Pass ints to specify a minimum and maximum allowed interaction
            range, in
            bin units.
        upper : bool
            Pass True to restrict the mask to the upper triangular entries.
        nan : bool
            Pass True to restrict the mask to non-NaN points.

        Returns
        -------
        np.ndarray
            The mask. Same shape as ``matrix``, with bool dtype.
        """
        mask = np.ones_like(matrix, dtype=bool)
        if min_range is not None:
            mask[np.triu(
                np.tril(np.ones_like(matrix, dtype=bool), k=min_range - 1),
                k=-(min_range - 1))] = False
        if max_range is not None:
            mask[np.triu(np.ones_like(matrix, dtype=bool),
                         k=max_range + 1)] = False
            mask[np.tril(np.ones_like(matrix, dtype=bool),
                         k=-(max_range + 1))] = False
        if upper:
            mask[np.tril(np.ones_like(matrix, dtype=bool), k=-1)] = False
        if nan:
            mask[np.isnan(matrix)] = False
        return mask

    @classmethod
    def apply(cls, f, *args, **kwargs):
        """
        Applies a function that takes in raw banded matrices (rectangular dense
        arrays) and returns raw banded matrices to one or more BandedMatrices.

        Parameters
        ----------
        f : function
            The function to apply. At least its first positional arg (or as many
            as all of them) should be an ``np.ndarray`` corresponding to the
            ``data`` array of a BandedMatrix. It should return one
            ``np.ndarray`` or a tuple of ``np.ndarray`` corresponding to the
            ``data`` array(s) of the resulting BandedMatrix(ces).
        *args : positional arguments
            Passed through to ``f()``. Instances of BandedMatrix will be passed
            as just their ``data`` arrays. If multiple BandedMatrices are
            passed, no attempt will be made to align them - a ValueError will be
            raised if they are not aligned.
        **kwargs : keyword arguments
            Passed through to ``f()``. No conversion from BandedMatrix will be
            attempted.

        Returns
        -------
        BandedMatrix or tuple of BandedMatrix
            If ``f()`` returns a single ``np.ndarray``, ``BandedMatrix.apply(f,
            arg1, ...)`` will return one BandedMatrix, whose shape and offsets
            will be taken from ``arg1`` (which must always be an BandedMatrix)
            and whose data will be taken from ``f(arg1.data, ...)``. If ``f()``
            returns a tuple of ``np.ndarray``, they will each be repackaged into
            an BandedMatrix in this fashion, and the tuple of new BandedMatrix
            will be returned.
        """
        if not all(np.array_equal(args[0].offsets, args[i].offsets)
                   for i in range(1, len(args)) if isinstance(args[i], cls)):
            raise ValueError('inputs not aligned')
        res_data = f(*(arg.data if isinstance(arg, cls) else arg
                       for arg in args), **kwargs)
        if isinstance(res_data, np.ndarray):
            return cls((res_data, args[0].offsets), shape=args[0].shape)
        return (cls((d, args[0].offsets), shape=args[0].shape)
                for d in res_data)

    @classmethod
    def max(cls, *matrices):
        """
        Returns the element-wise maximum of a list of BandedMatrices.

        Parameters
        ----------
        *matrices : sequence of BandedMatrix
            The matrices to compute the maximum of. Must already be aligned.

        Returns
        -------
        BandedMatrix
            The maximum.

        Notes
        -----
        If the matrices aren't aligned, consider:

            BandedMatrix.max(BandedMatrix.align(bm_a, bm_b, bm_c, ...))

        """
        return cls.apply(lambda *x: np.max(x, axis=0), *matrices)

    def __eq__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data == other.data
        else:
            res.data = res.data == other
        return res

    def __lt__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data < other.data
        else:
            res.data = res.data < other
        return res

    def __gt__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data > other.data
        else:
            res.data = res.data > other
        return res

    def __le__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data <= other.data
        else:
            res.data = res.data <= other
        return res

    def __ge__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data >= other.data
        else:
            res.data = res.data >= other
        return res

    def __and__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data & other.data
        else:
            raise NotImplementedError('type coercion not implemented yet')
        return res

    def __or__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data | other.data
        else:
            raise NotImplementedError('type coercion not implemented yet')
        return res

    def __xor__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
            res.data = res.data ^ other.data
        else:
            raise NotImplementedError('type coercion not implemented yet')
        return res

    def __invert__(self):
        res = self.copy()
        res.data = ~res.data
        return res

    def __add__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
        res.data += other
        return res

    def __sub__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
        res.data -= other
        return res

    def __mul__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
        res.data *= other
        return res

    def __div__(self, other):
        res = self.copy()
        if isinstance(other, sparse.dia_matrix):
            self.align(res, other)
        res.data /= other
        return res

    def __pow__(self, power, modulo=None):
        res = self.copy()
        res.data = res.data**power
        return res

    def log(self):
        res = self.copy()
        res.data = np.log(res.data)
        return res

    def flatten(self):
        return self.data.flatten()

    def ravel(self):
        return self.data.ravel()

    def where(self, a, b):
        cond = self.copy()
        cond.data = cond.data.astype(bool)
        res = self.copy()
        res.data = res.data.astype(np.promote_types(
            a.data.dtype if isinstance(a, sparse.dia_matrix) else type(a),
            b.data.dtype if isinstance(b, sparse.dia_matrix) else type(b)
        ))
        if isinstance(a, sparse.dia_matrix):
            self.align(res, a)
            res.data[cond.data] = a.data[cond.data]
        else:
            res.data[cond.data] = a
        if isinstance(b, sparse.dia_matrix):
            self.align(res, b)
            res.data[~cond.data] = b.data[~cond.data]
        else:
            res.data[~cond.data] = b
        return res

    def data_indices(self, key):
        # allow indexing with boolean BandedMatrix e.g., bm[bm > 10]
        if isinstance(key, BandedMatrix):
            if key.data.dtype != np.bool:
                raise IndexError('if indexing with BandedMatrix, dtype must '
                                 'be bool')
            return key.data

        # catch invalid key
        if not (type(key) in [tuple, list] and len(key) == 2):
            raise IndexError('must use tuple of length 2 to index into matrix')

        # correct trivial slices (to work with ogrid later)
        rowslice = slice(None, self.shape[0], None) \
            if type(key[0]) == slice and key[0] == slice(None, None, None) \
            else key[0]
        colslice = slice(None, self.shape[1], None) \
            if type(key[1]) == slice and key[1] == slice(None, None, None) \
            else key[1]

        if type(key[0]) == slice and type(key[1]) == slice:
            # if both items are slices then we just use ogrid on both
            row_idx, col_idx = np.ogrid[rowslice, colslice]
        elif type(key[1]) == slice:
            # column index is a slice
            col_idx = np.ogrid[colslice][np.newaxis, :]
            # set of rows
            if hasattr(key[0], '__len__'):
                row_idx = np.array([key[0]]).T
            # single row, presumably
            else:
                row_idx = key[0]
        elif type(key[0]) == slice:
            # row index is a slice
            row_idx = np.ogrid[rowslice][:, np.newaxis]
            # set of cols
            if hasattr(key[1], '__len__'):
                col_idx = np.array([key[1]])
            # single col, presumably
            else:
                col_idx = key[1]
        else:
            # arbitrary element (includes tuple of array)
            row_idx = np.asarray(key[0])
            col_idx = np.asarray(key[1])

        row_idx = (col_idx - row_idx + np.sum(self.offsets < 0))
        row_idx, col_idx = np.unravel_index(np.ravel_multi_index(
            (row_idx, col_idx), self.data.shape, mode='clip'), self.data.shape)
        return row_idx, col_idx

    def __getitem__(self, item):
        return self.data[self.data_indices(item)]

    def __setitem__(self, key, value):
        self.data[self.data_indices(key)] = value

    def reshape(self, shape, order='C'):
        raise NotImplementedError()

    def make_upper(self, pad=True):
        removed_diags = np.where(self.offsets < 0)[0]
        self.data = np.delete(self.data, removed_diags, axis=0)
        self.offsets = np.delete(self.offsets, removed_diags)
        if pad:
            self.data = np.pad(self.data, ((1, 0), (0, 0)), mode='constant',
                               constant_values=np.nan)
            self.offsets = np.pad(self.offsets, ((1, 0), ), mode='constant',
                                  constant_values=-1)

    def symmetrize(self):
        if np.sum(self.offsets < 0):
            self.make_upper(pad=False)
        assert self.offsets[0] == 0
        row_idx, col_idx = np.ogrid[:self.data.shape[0]-1, :self.data.shape[1]]
        row_idx = self.data.shape[0] - row_idx - 1
        col_idx = col_idx + row_idx
        row_idx, col_idx = np.unravel_index(np.ravel_multi_index(
            (row_idx, col_idx), self.data.shape, mode='wrap'), self.data.shape)
        self.data = np.concatenate([self.data[row_idx, col_idx], self.data])
        self.offsets = np.concatenate([-self.offsets[:0:-1], self.offsets])
        return self

    def deconvolute(self, bias, invert=False):
        """
        Applies a bias vector to both the rows and the columns of the
        BandedMatrix.

        Parameters
        ----------
        bias : np.ndarray
            Bias vector to apply. Should match size of BandedMatrix.
        invert : bool
            Pass True to invert the bias vector (divide the BandedMatrix by the
            the outer product of the bias vector with itself). Pass False to
            multiply the BandedMatrix by the outer product of the bias vector
            with itself.

        Returns
        -------
        BandedMatrix
            The result of the deconvolution.

        Examples
        --------
        >>> import numpy as np
        >>> from hic3defdr.util.banded_matrix import BandedMatrix
        >>> a = np.arange(16).reshape(4, 4).astype(float)
        >>> a += a.T
        >>> bias = np.sqrt(np.sum(a, axis=0))
        >>> b = ((a / bias).T / bias).T
        >>> bm = BandedMatrix.from_ndarray(b, max_range=3)
        >>> bm = bm.deconvolute(bias)
        >>> np.allclose(bm.toarray(), a)
        True
        """
        if invert:
            bias = 1 / bias
        csr = self.tocsr()
        bias_csr = sparse.diags([bias], [0])
        biased_csr = bias_csr.dot(csr)
        biased_csr = biased_csr.dot(bias_csr)
        return self.from_sparse(
            biased_csr, max_range=self.max_range(), upper=self.is_upper())
