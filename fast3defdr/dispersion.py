import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from lib5c.util.lowess import lowess_fit

from fast3defdr.scaled_nb import inverse_mvr
from fast3defdr.binning import equal_bin


def cml(data):
    """
    Estimates the dispersion parameter of a NB distribution given the data via
    conditional maximum likelihood.

    One common dispersion will be estimated across all pixels in the data,
    though the individual pixels in the data may have distinct means.

    Parameters
    ----------
    data : np.ndarray
        Rows should correspond to pixels, columns to replicates.

    Returns
    -------
    float
        The estimated dispersion.
    """
    n = data.shape[1]
    z = np.sum(data, axis=1)

    def nll(delta):
        r = 1./delta - 1
        return -np.sum((np.sum(gammaln(data + r), axis=1) + gammaln(n*r) -
                        gammaln(z+n*r) - n * gammaln(r)))

    res = minimize_scalar(nll, bounds=(1e-4, 100./(100+1)), method='bounded')
    assert res.success
    delta_hat = res.x
    return delta_hat / (1-delta_hat)


def mme_per_pixel(data):
    """
    Estimates the dispersion parameter of a separate NB distribution for each
    pixel given the data using a method of moments approach.

    Parameters
    ----------
    data : np.ndarray
        Rows should correspond to pixels, columns to replicates.

    Returns
    -------
    np.ndarray
        The estimated dispersion for each pixel.
    """
    m = np.mean(data, axis=1)
    v = np.var(data, axis=1, ddof=1)
    return inverse_mvr(m, v)


def mme(data):
    """
    Estimates the dispersion parameter of a NB distribution given the data using
    a method of moments approach.

    One common dispersion will be estimated across all pixels in the data,
    though the individual pixels in the data may have distinct means.

    Parameters
    ----------
    data : np.ndarray
        Rows should correspond to pixels, columns to replicates.

    Returns
    -------
    float
        The estimated dispersion.
    """
    return np.nanmean(mme_per_pixel(data))


def estimate_dispersion(data, cov, estimator='cml', n_bins=100, logx=True):
    """
    Estimates trended dispersion for each point in ``data`` with respect to a
    covariate ``cov``, using ``estimator`` to estimate the dispersion within
    each of ``n_bins`` equal-number bins and fitting a curve through the per-bin
    dispersion estimates using lowess smoothing.

    Parameters
    ----------
    data : np.ndarray
        Rows should correspond to pixels, columns to replicates.
    cov : np.ndarray
        A vector of covariate values per pixel.
    estimator : 'cml', 'mme', or a function
        Pass 'cml' or 'mme' to use conditional maximum likelihood or method of
        moments estimation to estimate the dispersion within each bin. Pass a
        function that takes in a (pixels, replicates) shaped array of data and
        returns a dispersion value to use that instead.
    n_bins : int
        The number of bins to use when binning the pixels according to ``cov``.
    logx : bool
        Whether or not to perform the lowess fit in log x space.

    Returns
    -------
    smoothed_disp : np.ndarray
        Vector of smoothed dispersion estimates for each pixel.
    cov_per_bin, disp_per_bin : np.ndarray
        Average covariate value and estimated dispersion value, respectively,
        per bin.
    """
    if type(estimator) == str and estimator not in {'cml', 'mme'}:
        raise ValueError('estimator must be \'cml\', \'mme\' or a function')
    disp_func = globals()[estimator] if estimator in globals().keys() \
        else estimator
    bins = equal_bin(cov, n_bins)
    cov_per_bin = np.array([np.mean(cov[bins == b]) for b in range(n_bins)])
    disp_per_bin = np.array([disp_func(data[bins == b, :])
                             for b in range(n_bins)])
    cov_idx = cov_per_bin > 0 if logx is True \
        else np.ones_like(cov_per_bin, dtype=bool)
    disp_smooth_func = lowess_fit(cov_per_bin[cov_idx], disp_per_bin[cov_idx],
                                  left_boundary=None, logx=logx, logy=True)
    smoothed_disp = disp_smooth_func(cov)
    return smoothed_disp, cov_per_bin, disp_per_bin
