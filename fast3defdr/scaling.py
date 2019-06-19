import numpy as np

from lib5c.util.mathematics import gmean


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
