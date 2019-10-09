import numpy as np
import scipy.stats as stats

from hic3defdr.scaled_nb import logpmf, fit_mu_hat


def lrt(raw, f, disp, design, refit_mu=True):
    """
    Performs a likelihood ratio test on raw data ``raw`` given scaling factors
    ``f`` and dispersion ``disp``.

    Parameters
    ----------
    raw, f, disp : np.ndarray
        Matrices of raw values, combined scaling factors, and dispersions,
        respectively. Rows correspond to pixels, columns correspond to
        replicates.
    design : np.ndarray
        Describes the grouping of replicates into conditions. Rows correspond to
        replicates, columns correspond to conditions, and values should be True
        where a replicate belongs to a condition and False otherwise.

    Returns
    -------
    pvalues : np.ndarray
        The LRT p-values per pixel.
    llr : np.ndarray
        The log likelihood ratio per pixel.
    mu_hat_null, mu_hat_alt : np.ndarray
        The fitted mean parameters under the null and alt models, respectively,
        per pixel.
    """
    if refit_mu:
        mu_hat_null = fit_mu_hat(raw, f, disp)
        mu_hat_alt = np.array(
            [fit_mu_hat(raw[:, design[:, c]],
                        f[:, design[:, c]],
                        disp[:, design[:, c]])
             for c in range(design.shape[1])]).T
    else:
        mu_hat_null = np.mean(raw / f, axis=1)
        mu_hat_alt = np.array(
            [np.mean(raw[:, design[:, c]] / f[:, design[:, c]], axis=1)
             for c in range(design.shape[1])]).T
    mu_hat_alt_wide = np.dot(mu_hat_alt, design.T)
    null_ll = np.sum(logpmf(raw, mu_hat_null[:, None] * f, disp), axis=1)
    alt_ll = np.sum(logpmf(raw, mu_hat_alt_wide * f, disp), axis=1)
    llr = null_ll - alt_ll
    pvalues = stats.chi2(design.shape[1] - 1).sf(-2 * llr)
    return pvalues, llr, mu_hat_null, mu_hat_alt
