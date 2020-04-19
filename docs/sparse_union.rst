Using ``sparse_union()`` to load a rectangular data matrix
==========================================================

Motivation
----------

The hic3defdr ecosystem revolves around a :doc:`specific data layout <data_layout>`.
hic3defdr automatically imports your input npz's into this layout when running
the pipeline. But what if you want to convert your npz's into hic3efdr's data
layout without running the pipeline?

The key ingredient that will allow us to do this is the function
:py:func:`hic3defdr.util.matrices.sparse_union()`, which will be demonstrated
below.

Walkthrough
-----------

In an interactive shell, import
:py:func:`hic3defdr.util.matrices.sparse_union()`:

    >>> import numpy as np
    >>> import scipy.sparse as sparse
    >>> from hic3defdr.util.matrices import sparse_union

The kwargs on this function mostly serve to help filter out pixels to reduce the
total number of pixels. The most important kwarg for this is ``dist_thresh``,
which simply drops all pixels with interaction distances longer than
``dist_thresh``. This threshold is on by default and is highly recommended.

The remaining kwargs work together to throw out pixels that have low mean values
across replicates after normalization (by ``bias`` and ``size_factors`` if these
are passed). This filtering is off by default (``mean_thresh=0.0``) and is not
recommended. This means that you don't need to worry about passing `bias` or
``size_factors`` to this function even if you have these values available.

Create some test npz's from dense matrices with zeros in different positions:

    >>> rep1 = np.array([[0., 0., 3., 1.],
    ...                  [0., 6., 5., 0.],
    ...                  [0., 0., 0., 2.],
    ...                  [0., 0., 0., 7.]])
    >>> rep2 = np.array([[0., 1., 3., 2.],
    ...                  [0., 0., 0., 0.],
    ...                  [0., 0., 4., 2.],
    ...                  [0., 0., 0., 3.]])
    >>> sparse.save_npz('rep1.npz', sparse.csr_matrix(rep1))
    >>> sparse.save_npz('rep2.npz', sparse.csr_matrix(rep2))

Use ``sparse_union`` to identify the union pixel set:

    >>> rep_npzs = ['rep1.npz', 'rep2.npz']
    >>> row, col = sparse_union(rep_npzs, dist_thresh=2)
    >>> list(zip(row, col))
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]

Notice that pixels (0, 0) and (1, 3) are not in the union pixel set. This is
because these pixels are zero in both replicates.

Also notice that pixel (0, 3) is also not in the union pixel set. This is
because its distance (3 - 0 = 3) is greater than the ``dist_thresh`` we passed
to ``sparse_union()``.

Finally, we can construct the ``data`` matrix:

    >>> data = np.zeros((len(row), len(rep_npzs)))
    >>> for i in range(len(rep_npzs)):
    ...     data[:, i] = sparse.load_npz(rep_npzs[i]).tocsr()[row, col]
    >>> data
    array([[0., 1.],
           [3., 3.],
           [6., 0.],
           [5., 0.],
           [0., 4.],
           [2., 2.],
           [7., 3.]])

If we want to know the interaction distance for each pixel (each row of
``data``), we can calculate:

    >>> dist = col - row
    >>> dist
    array([1, 2, 0, 1, 0, 1, 0], dtype=int32)

To clean up, we will delete the npz's we created.:

    >>> import os
    >>> for f in rep_npzs:
    ...     os.remove(f)
