import numpy as np
import scipy.stats as stats
from scipy.special import gammaln
from scipy.optimize import newton, brentq

from lib5c.util.mathematics import gmean

from hic3defdr.logging import eprint
from hic3defdr.progress import tqdm_maybe as tqdm


def logpmf(k, m, phi):
    """
    Log of the PMF of the negative binomial distribution, parameterized by its
    mean ``m`` and dispersion ``phi``. Vectorized.

    Parameters
    ----------
    k : int
        The number of counts observed.
    m : float
        The mean parameter.
    phi : float
        The dispersion parameter.

    Returns
    -------
    float
        The log of the probability of observing ``k`` counts.
    """
    r = 1. / phi
    return gammaln(r+k) - gammaln(k+1) - gammaln(r) + \
        r*np.log(r) - r*np.log(r+m) + k*np.log(m) - k*np.log(r+m)


def mvr(mean, disp):
    """
    Negative binomial fixed-dispersion mean-variance relationship. Vectorized.

    Parameters
    ----------
    mean, disp : float
        The mean and dispersion of a NB distribution, respectively.

    Returns
    -------
    float
        The variance of that NB distribution.
    """
    return mean + mean**2 * disp


def inverse_mvr(mean, var):
    """
    Inverse function of the negative binomial fixed-dispersion mean-variance
    relationship. Vectorized.

    Parameters
    ----------
    mean, var : float
        The mean and variance of a NB distribution, respectively.

    Returns
    -------
    float
        The dispersion of that NB distribution.
    """
    return (var-mean) / mean**2


def fit_mu_hat(x, b, alpha, verbose=True):
    """
    Numerical MLE fitter for the mean parameter of the scaled NB model under
    fixed dispersion. Vectorized.

    See the following colab notebook for background and derivation:
    https://colab.research.google.com/drive/1SgMMvc3XhfIXoBx8tsyJt-yyBDlRFQCJ

    Parameters
    ----------
    x : np.ndarray
        The vector of observed counts.
    b : np.ndarray
        The vector of scaling factors, parallel to ``x``.
    alpha : np.ndarray
        The vector of dispersions, parallel to ``x``.
    verbose : bool
        Pass False to silence reporting of progress to stderr.

    Returns
    -------
    float
        The MLE of the mean parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from hic3defdr.scaled_nb import fit_mu_hat

    3 pixels, 2 reps (matrices):
    >>> x = np.array([[1, 2],
    ...               [3, 4],
    ...               [5, 6]])
    >>> b = np.array([[0.9, 1.1],
    ...               [0.8, 1.2],
    ...               [0.7, 1.3]])
    >>> alpha = np.array([[0.1, 0.2],
    ...                   [0.3, 0.4],
    ...                   [0.5, 0.6]])
    >>> fit_mu_hat(x, b, alpha)
    array([1.47251127, 3.53879843, 5.86853465])

    broadcast dispersion down the pixels:
    >>> fit_mu_hat(x, b, np.array([0.1, 0.2]))
    array([1.47251127, 3.53749833, 5.85554075])

    broadcast dispersion across the reps:
    >>> fit_mu_hat(x, b, np.array([0.1, 0.2, 0.3])[:, None])
    array([1.49544092, 3.51679438, 5.73129492])

    1 pixel, two reps (vectors):
    >>> fit_mu_hat(np.array([1, 2]), np.array([0.9, 1.1]), np.array([0.1, 0.2]))
    array([1.47251127])

    broadcast dispersion across reps:
    >>> fit_mu_hat(np.array([1, 2]), np.array([0.9, 1.1]), 0.1)
    array([1.49544092])

    one pixel is fitted with newton, the second is fitted with brentq
    >>> x = np.array([[2, 3, 4, 2],
    ...               [6, 9, 3, 1]])
    >>> b = np.array([[0.45, 0.53, 0.088, 0.091],
    ...               [0.70, 0.83, 0.14,  0.15 ]])
    >>> alpha = np.array([[0.0071, 0.0071, 0.0073, 0.0073],
    ...                   [0.0070, 0.0070, 0.0072, 0.0072]])
    >>> fit_mu_hat(x, b, alpha)
    array([ 9.5900971 , 10.45962955])
    """
    assert np.all((alpha > 0) & np.isfinite(alpha))
    assert np.all((x >= 0) & np.isfinite(x))
    assert np.all((b > 0) & np.isfinite(b))

    def f(mu_hat):
        if hasattr(mu_hat, 'ndim') and mu_hat.ndim < b.ndim and mu_hat.ndim > 0:
            mu_hat = mu_hat[:, None]
        return np.sum((x - mu_hat*b) / (mu_hat + alpha * mu_hat**2 * b),
                      axis=-1)

    if not x.ndim == 2:
        # only one pixel, no need to parallelize
        root = np.array([-1.0])
        failed = np.array([True])
    else:
        # multiple pixels, use newton
        root, converged, zero_der = newton(
            f, np.mean(x / b, axis=1), maxiter=100, full_output=True)
        failed = ~converged | zero_der
        failed[root <= 0] = True  # fail points with negative mu
        failed[root >= np.sqrt(np.finfo(float).max) / 1e10] = True  # overflow
        failed[~np.isclose(f(root), 0, atol=1e-5)] = True  # these aren't close

    if np.any(failed):
        eprint('some points failed, fixing with brentq',
               skip=x.ndim != 2 or not verbose)
        for idx in tqdm(np.where(failed)[0],
                        disable=x.ndim != 2 or not verbose):
            lower = 10 * np.finfo(float).eps
            upper = np.mean(x[idx] / b[idx])
            counter = 0
            while True:
                try:
                    root[idx] = brentq(
                        f if np.isscalar(f(lower)) else lambda y: f(y)[idx],
                        lower, upper)
                    break
                except ValueError:
                    upper *= 2
                    counter += 1
                    if counter > 100:
                        raise ValueError('bracketing interval not found within '
                                         '100 doublings')
    assert np.allclose(f(root), 0, atol=1e-5)
    return root


def equalize(data, f, alpha):
    """
    Given known scaling factors ``f`` and a known dispersion ``alpha``, creates
    common-scale pseudodata from raw values ``data``.

    See https://rdrr.io/bioc/edgeR/src/R/equalizeLibSizes.R

    Parameters
    ----------
    data : np.ndarray
        Matrix of raw data to equalize. Rows are pixels, columns are replicates.
    f : np.ndarray
        Matrix of combined scaling factors for each pixel.
    alpha : float
       Single fixed dispersion to use during equalization.

    Returns
    -------
    np.ndarray
        Matrix of equalized data.
    """
    f_mean = gmean(f, pseudocount=0, axis=1)
    mu_hat = fit_mu_hat(data, f, alpha)
    mu_in = mu_hat[:, None] * f
    mu_out = mu_hat * f_mean
    pseudodata = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        pseudodata[:, i] = q2qnbinom(data[:, i], mu_in[:, i], mu_out, alpha)
    return pseudodata


def q2qnbinom(x, mu_in, mu_out, alpha):
    """
    Converts values between two NB distributions with different means but the
    same dispersion.

    See https://rdrr.io/bioc/edgeR/src/R/q2qnbinom.R

    Parameters
    ----------
    x : np.ndarray
        Vector of values to convert.
    mu_in, mu_out : np.ndarray
        Vectors of means to convert between.
    alpha : np.ndarray or float
        Single dispserion (to use for all ``x``) or vector of dispersions (one
        per ``x``) to hold constant during conversion.

    Returns
    -------
    np.ndarray
        ``x`` converted from ``mu_in`` to ``mu_out``.
    """
    # force minimum mu of 0.25 for stability
    high_idx = (mu_in >= 0.25) & (mu_out >= 0.25)
    mu_in[~high_idx] = 0.25
    mu_out[~high_idx] = 0.25

    # compute fano factors
    r_in = 1 + alpha * mu_in
    r_out = 1 + alpha * mu_out

    # compute variances
    v_in = mu_in * r_in
    v_out = mu_out * r_out

    # mark points as left or right tail
    right_idx = x >= mu_in

    # construct normal and gamma distributions
    norm_in = stats.norm(mu_in, np.sqrt(v_in))
    norm_out = stats.norm(mu_out, np.sqrt(v_out))
    gamma_in = stats.gamma(mu_in/r_in, scale=r_in)
    gamma_out = stats.gamma(mu_out/r_out, scale=r_out)

    # convert left and right tail separately
    q_norm = np.zeros_like(mu_in)
    q_gamma = np.zeros_like(mu_in)
    q_norm[right_idx] = norm_out.isf(norm_in.sf(x))[right_idx]
    q_norm[~right_idx] = norm_out.ppf(norm_in.cdf(x))[~right_idx]
    q_gamma[right_idx] = gamma_out.isf(gamma_in.sf(x))[right_idx]
    q_gamma[~right_idx] = gamma_out.ppf(gamma_in.cdf(x))[~right_idx]

    # compute pseudocounts as average
    pseudocounts = (q_norm + q_gamma) / 2

    # clip to zero
    pseudocounts[~(pseudocounts >= 0)] = 0

    return pseudocounts
