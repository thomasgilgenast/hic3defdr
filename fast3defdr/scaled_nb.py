import numpy as np
from scipy.special import gammaln
from scipy.optimize import newton, brentq


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


def fit_mu_hat(x, b, alpha):
    """
    Numerical MLE fitter for the mean parameter of the scaled NB model under
    fixed dispersion. Vectorized.

    Parameters
    ----------
    x : np.ndarray
        The vector of observed counts.
    b : np.ndarray
        The vector of scaling factors, parallel to ``x``.
    alpha : np.ndarray
        The vector of dispersions, parallel to ``x``.

    Returns
    -------
    float
        The MLE of the mean parameter.
    """
    print 'hi'
    assert np.all((alpha > 0) & np.isfinite(alpha))
    assert np.all((x >= 0) & np.isfinite(x))
    assert np.all((b > 0) & np.isfinite(b))

    def f(mu_hat):
        if hasattr(mu_hat, 'ndim') and mu_hat.ndim < b.ndim:
            mu_hat = mu_hat[:, None]
        return np.sum((x - mu_hat*b) / (mu_hat + alpha * mu_hat**2 * b), axis=1)
    root, converged, zero_der = newton(f, np.mean(x / b, axis=1), maxiter=1000,
                                       full_output=True)
    failed = ~converged | zero_der
    failed[root <= 0] = True  # fail points with negative mu
    failed[root >= np.sqrt(np.finfo(float).max) / 1e10] = True  # these overflow
    failed[~np.isclose(f(root), 0)] = True  # these aren't close
    for idx in np.where(failed)[0]:
        print('fixing failed point %s/%s' % (idx, failed.sum()))
        lower = 10 * np.finfo(float).eps
        upper = np.mean(x[idx] / b[idx])
        counter = 0
        while True:
            try:
                root[idx] = brentq(lambda x: f(x)[idx], lower, upper)
                break
            except ValueError:
                upper *= 2
                counter += 1
                if counter > 100:
                    raise ValueError('bracketing interval not found within '
                                     '100 doublings')
    assert np.allclose(f(root), 0, atol=1e-5)
    return root
