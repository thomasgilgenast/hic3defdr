import numpy as np

from lib5c.util.mathematics import gmean

from fast3defdr.binning import equal_bin
from fast3defdr.progress import tqdm_maybe as tqdm


def median_of_ratios(data):
    """
    Computes size factors for a dataset using the median of ratios method.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.

    Returns
    -------
    np.ndarray
        Vector of size factors, per replicate.
    """
    return np.median(data / gmean(data, axis=1)[:, None], axis=0)


def simple_scaling(data):
    """
    Computes size factors for a dataset using a simple scaling method.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.

    Returns
    -------
    np.ndarray
        Vector of size factors, per replicate.
    """
    s = np.sum(data, axis=0)
    return s / gmean(s)


def conditional(data, dist, fn, idx=None, n_bins=None):
    """
    Applies a size factor computing function ``fn`` to ``data`` conditioning on
    ``dist``, optionally applying ``idx`` to filter inputs to ``fn`` and/or
    binning ``dist`` into ``n_bins`` equal-number bins.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.
    dist : np.ndarray
        The distance of each pixel in ``data``.
    fn : function
        A function that computes a size factor given some data.
    idx : np.ndarray, optional
        A boolean vector indicating whether or not each pixel in ``data`` should
        be included in what's passed to ``fn``.
    n_bins : int, optional
        Pass an int to bin distance into this many equal-number bins.
    """
    if idx is None:
        idx = np.ones_like(dist, dtype=bool)
    result = np.zeros_like(data, dtype=float)
    if n_bins:
        bins = equal_bin(dist, n_bins)
    else:
        bins = dist
    for b in tqdm(np.unique(bins)):
        dist_idx = bins == b
        result[dist_idx, :] = fn(data[dist_idx & idx, :])
    return result


def conditional_mor(data, dist, n_bins=None):
    """
    Computes size factors for a dataset using median of ratios normalization,
    conditioning on distance.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.
    dist : np.ndarray
        The distance of each pixel in ``data``
    n_bins : int, optional
        Pass an int to bin distance into this many equal-number bins.

    Returns
    -------
    np.ndarray
        Matrix of size factors, per pixel (rows) and per replicate (columns).
    """
    zero_idx = np.all(data > 0, axis=1)
    return conditional(data, dist, median_of_ratios, idx=zero_idx,
                       n_bins=n_bins)


def conditional_scaling(data, dist, n_bins=None):
    """
    Computes size factors for a dataset using simple scaling normalization,
    conditioning on distance.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.
    dist : np.ndarray
        The distance of each pixel in ``data``.
    n_bins : int, optional
        Pass an int to bin distance into this many equal-number bins.

    Returns
    -------
    np.ndarray
        Matrix of size factors, per pixel (rows) and per replicate (columns).
    """
    return conditional(data, dist, simple_scaling, n_bins=n_bins)
