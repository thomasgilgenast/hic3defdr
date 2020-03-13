import numpy as np


def equal_bin(data, n_bins):
    """
    Bins ``data`` into ``n_bins`` bins, with an equal number of points in each
    bin.

    https://stackoverflow.com/a/40895507

    Parameters
    ----------
    data : np.ndarray
        The data to bin.
    n_bins : int
        The number of bins to bin into.

    Returns
    -------
    np.ndarray
        A vector of integers representing the bin index for each entry in
        ``data``.
    """
    idx = np.linspace(0, n_bins, data.size, endpoint=0, dtype=int)
    return idx[data.argsort().argsort()]
