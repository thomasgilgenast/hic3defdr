import numpy as np
from scipy.interpolate import interp1d

from lib5c.util.mathematics import gmean

from hic3defdr.binning import equal_bin
from hic3defdr.progress import tqdm_maybe as tqdm


def no_scaling(data):
    """
    Computes dummy size factors of 1 for each replicate.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.

    Returns
    -------
    np.ndarray
        Vector of size factors, per replicate.
    """
    return np.ones(data.shape[1], dtype=float)


def median_of_ratios(data, filter_zeros=True):
    """
    Computes size factors for a dataset using the median of ratios method.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.
    filter_zeros : bool
        Pass True to filter out pixels with a zero value in any replicate. Pass
        False to include all data.

    Returns
    -------
    np.ndarray
        Vector of size factors, per replicate.
    """
    idx = np.all(data > 0, axis=1) if filter_zeros \
        else np.ones(data.shape[0], dtype=bool)
    return np.median(data[idx, :] /
                     gmean(data[idx, :], axis=1)[:, None], axis=0)


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


def conditional(data, dist, fn, n_bins=None):
    """
    Applies a size factor computing function ``fn`` to ``data`` conditioning on
    ``dist``, optionally binning ``dist`` into ``n_bins`` equal-number bins.

    If ``n_bins`` is not None, the final size factors will be interpolated for
    interaction distances falling between the bins to avoid bin edge effects.

    Parameters
    ----------
    data : np.ndarray
        Rows correspond to pixels, columns correspond to replicates.
    dist : np.ndarray
        The distance of each pixel in ``data``.
    fn : function
        A function that computes a size factor given some data.
    n_bins : int, optional
        Pass an int to bin distance into this many equal-number bins.
    """
    result = np.zeros_like(data, dtype=float)
    if n_bins:
        bins = equal_bin(dist, n_bins)
        d_per_bin = []
        s_per_bin = []
        for b in tqdm(np.unique(bins)):
            dist_idx = bins == b
            d_per_bin.append(np.mean(dist[dist_idx]))
            s_per_bin.append(fn(data[dist_idx, :]))
        interp_fns = [interp1d(np.array(d_per_bin), np.array(s_per_bin)[:, i],
                               fill_value='extrapolate', assume_sorted=True)
                      for i in range(data.shape[1])]
        for i in range(data.shape[1]):
            result[:, i] = interp_fns[i](dist)
    else:
        for d in tqdm(np.unique(dist)):
            dist_idx = dist == d
            result[dist_idx, :] = fn(data[dist_idx, :])
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
    return conditional(data, dist, median_of_ratios, n_bins=n_bins)


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
