"""
Experimental module exposing variants of the HiC3DeFDR model for benchmarking
purposes.
"""

import numpy as np
import scipy.stats as stats

from hic3defdr.analysis import HiC3DeFDR
from hic3defdr.util.printing import eprint
from hic3defdr.util.dispersion import mme_per_pixel
from hic3defdr.util.clusters import load_clusters
from hic3defdr.util.parallelization import parallel_apply
import hic3defdr.util.dispersion as dispersion


def poisson_fit_mu_hat(raw, f):
    return np.average(raw / f, weights=f, axis=1)


def poisson_logpmf(x, mu):
    return stats.poisson(mu).logpmf(x)


def poisson_lrt(raw, f, design, refit_mu=True):
    if refit_mu:
        mu_hat_null = poisson_fit_mu_hat(raw, f)
        mu_hat_alt = np.array(
            [poisson_fit_mu_hat(raw[:, design[:, c]], f[:, design[:, c]])
             for c in range(design.shape[1])]
        ).T
    else:
        mu_hat_null = np.mean(raw / f, axis=1)
        mu_hat_alt = np.array(
            [np.mean(raw[:, design[:, c]] / f[:, design[:, c]], axis=1)
             for c in range(design.shape[1])])
    mu_hat_alt_wide = np.dot(mu_hat_alt, design.T)
    null_ll = np.sum(poisson_logpmf(raw, mu_hat_null[:, None] * f), axis=1)
    alt_ll = np.sum(poisson_logpmf(raw, mu_hat_alt_wide * f), axis=1)
    llr = null_ll - alt_ll
    pvalues = stats.chi2(design.shape[1] - 1).sf(-2 * llr)
    return pvalues, llr, mu_hat_null, mu_hat_alt


class Poisson3DeFDR(HiC3DeFDR):
    def estimate_disp(self, estimator='qcml', frac=None, auto_frac_factor=15.,
                      weighted_lowess=True, n_threads=-1):
        # note: all kwargs are ignored
        eprint('estimating dispersion')
        estimator = dispersion.__dict__[estimator] \
            if estimator in dispersion.__dict__ else estimator

        eprint('  loading data')
        disp_idx, _ = self.load_data('disp_idx', 'all')
        row, offsets = self.load_data('row', 'all', idx=disp_idx)
        col, _ = self.load_data('col', 'all', idx=disp_idx)
        scaled, _ = self.load_data('scaled', 'all', idx=disp_idx)

        eprint('  computing pixel-wise mean per condition')
        disp_per_dist = np.zeros((self.dist_thresh_max+1, self.design.shape[1]))
        disp = np.zeros((disp_idx.sum(), self.design.shape[1]))

        def disp_fn(mean):
            return np.zeros_like(mean)

        for c, cond in enumerate(self.design.columns):
            self.save_disp_fn(cond, disp_fn)

        eprint('  saving estimated dispersions to disk')
        self.save_data(disp, 'disp', offsets)
        self.save_data(disp_per_dist, 'disp_per_dist')

    def lrt(self, chrom=None, refit_mu=True, n_threads=-1, verbose=True):
        if chrom is None:
            if n_threads:
                parallel_apply(
                    self.lrt,
                    [{'chrom': c, 'refit_mu': refit_mu, 'verbose': False}
                     for c in self.chroms],
                    n_threads=n_threads
                )
            else:
                for chrom in self.chroms:
                    self.lrt(chrom=chrom, refit_mu=refit_mu)
            return
        eprint('running LRT for chrom %s' % chrom)
        eprint('  loading data', skip=not verbose)
        bias = self.load_bias(chrom)
        size_factors = self.load_data('size_factors', chrom)
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        raw = self.load_data('raw', chrom)
        disp_idx = self.load_data('disp_idx', chrom)

        eprint('  computing LRT results', skip=not verbose)
        f = bias[row, :][disp_idx, :] * bias[col, :][disp_idx, :] * \
            size_factors[disp_idx, :]
        pvalues, llr, mu_hat_null, mu_hat_alt = poisson_lrt(
            raw[disp_idx, :], f, self.design.values, refit_mu=True)

        if self.loop_patterns:
            eprint('  making loop_idx', skip=not verbose)
            loop_pixels = set().union(
                *sum((load_clusters(pattern.replace('<chrom>', chrom))
                      for pattern in self.loop_patterns.values()), []))
            loop_idx = np.array([True if pixel in loop_pixels else False
                                 for pixel in zip(row[disp_idx],
                                                  col[disp_idx])])
            self.save_data(loop_idx, 'loop_idx', chrom)

        eprint('  saving results to disk', skip=not verbose)
        self.save_data(pvalues, 'pvalues', chrom)
        self.save_data(llr, 'llr', chrom)
        self.save_data(mu_hat_null, 'mu_hat_null', chrom)
        self.save_data(mu_hat_alt, 'mu_hat_alt', chrom)


class Unsmoothed3DeFDR(HiC3DeFDR):
    def estimate_disp(self, estimator='qcml', frac=None, auto_frac_factor=15.,
                      weighted_lowess=True, n_threads=-1):
        # note: all kwargs are ignored
        eprint('estimating dispersion')
        eprint('  loading data')
        disp_idx, _ = self.load_data('disp_idx', 'all')
        row, offsets = self.load_data('row', 'all', idx=disp_idx)
        col, _ = self.load_data('col', 'all', idx=disp_idx)
        scaled, _ = self.load_data('scaled', 'all', idx=disp_idx)

        eprint('  computing pixel-wise mean per condition')
        disp = np.zeros((disp_idx.sum(), self.design.shape[1]))
        for c, cond in enumerate(self.design.columns):
            eprint('  estimating dispersion for condition %s' % cond)
            disp[:, c] = np.maximum(mme_per_pixel(
                scaled[:, self.design[cond]]), 1e-7)

        eprint('  saving estimated dispersions to disk')
        self.save_data(disp, 'disp', offsets)


class Global3DeFDR(HiC3DeFDR):
    def estimate_disp(self, estimator='qcml', frac=None, auto_frac_factor=15.,
                      weighted_lowess=True, n_threads=-1):
        # note: all kwargs except estimator are ignored
        eprint('estimating dispersion')
        estimator = dispersion.__dict__[estimator] \
            if estimator in dispersion.__dict__ else estimator

        eprint('  loading data')
        disp_idx, disp_idx_offsets = self.load_data('disp_idx', 'all')
        loop_idx, _ = self.load_data('loop_idx', 'all')
        row, offsets = self.load_data('row', 'all', idx=disp_idx)
        col, _ = self.load_data('col', 'all', idx=disp_idx)
        raw, _ = self.load_data('raw', 'all', idx=disp_idx)
        f = np.ones_like(raw, dtype=float)
        for i, chrom in enumerate(self.chroms):
            chrom_slice = slice(offsets[i], offsets[i+1])
            row_chrom = row[chrom_slice]
            col_chrom = col[chrom_slice]
            disp_idx_chrom = disp_idx[disp_idx_offsets[i]:disp_idx_offsets[i+1]]
            bias = self.load_bias(chrom)
            size_factors = self.load_data('size_factors', chrom)[disp_idx_chrom]
            f[chrom_slice] = bias[row_chrom, :] * bias[col_chrom, :] \
                * size_factors

        disp = np.zeros((disp_idx.sum(), self.design.shape[1]))
        disp_per_dist = np.zeros((self.dist_thresh_max+1, self.design.shape[1]))
        for c, cond in enumerate(self.design.columns):
            eprint('  estimating dispersion for condition %s' % cond)
            global_disp = estimator(raw[loop_idx, :][:, self.design[cond]],
                                    f=f[loop_idx, :][:, self.design[cond]])
            disp[:, c] = global_disp
            disp_per_dist[:, c] = global_disp

            def disp_fn(mean):
                return np.ones_like(mean) * global_disp

            self.save_disp_fn(cond, disp_fn)

        eprint('  saving estimated dispersions to disk')
        self.save_data(disp, 'disp', offsets)
        self.save_data(disp_per_dist, 'disp_per_dist')
