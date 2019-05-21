import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats

from lib5c.util.system import check_outdir
from lib5c.util.mathematics import gmean
from lib5c.util.statistics import adjust_pvalues
from hiclite.util.clusters import load_clusters, find_clusters, save_clusters

from fast3defdr.sparse_intersection import sparse_intersection
from fast3defdr.scaled_nb import logpmf, fit_mu_hat

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


class Fast3DeFDR(object):
    def __init__(self, raw_npz_patterns, bias_patterns, chroms, design, outdir,
                 dist_thresh_min=4, dist_thresh_max=1000, bias_thresh=0.1,
                 disp_thresh_mean=5, loop_patterns=None):
        self.raw_npz_patterns = raw_npz_patterns
        self.bias_patterns = bias_patterns
        self.chroms = chroms
        if type(design) == str:
            self.design = pd.read_csv(design, index_col=0)
        else:
            self.design = design
        self.outdir = outdir
        self.dist_thresh_min = dist_thresh_min
        self.dist_thresh_max = dist_thresh_max
        self.bias_thresh = bias_thresh
        self.disp_thresh_mean = disp_thresh_mean
        self.loop_patterns = loop_patterns
        self.picklefile = '%s/pickle' % self.outdir
        check_outdir(self.picklefile)
        with open(self.picklefile, 'wb') as handle:
            pickle.dump(self, handle, -1)

    @classmethod
    def load(cls, outdir):
        with open('%s/pickle' % outdir, 'rb') as handle:
            return pickle.load(handle)

    def process_chrom(self, chrom):
        print('processing chrom %s' % chrom)
        print('  loading bias')
        bias = np.array([np.loadtxt(pattern.replace('<chrom>', chrom))
                         for pattern in self.bias_patterns]).T

        print('  computing union pixel set')
        row, col = sparse_intersection([pattern.replace('<chrom>', chrom)
                                        for pattern in self.raw_npz_patterns])

        print('  pre-filtering')
        # compute dist_idx
        dist = col - row
        dist_idx = (dist >= self.dist_thresh_min) & \
                   (dist <= self.dist_thresh_max)

        # compute bias_idx
        bias_idx = \
            np.all(bias[row, :] >= self.bias_thresh, axis=1) & \
            np.all(bias[col, :] >= self.bias_thresh, axis=1) & \
            np.all(bias[row, :] <= 1/self.bias_thresh, axis=1) & \
            np.all(bias[col, :] <= 1/self.bias_thresh, axis=1)

        # apply dist_idx and bias_idx, overwriting row, col
        row = row[dist_idx & bias_idx]
        col = col[dist_idx & bias_idx]

        print('  loading raw data')
        raw = np.zeros((len(row), len(self.raw_npz_patterns)), dtype=int)
        for i, pattern in enumerate(self.raw_npz_patterns):
            raw[:, i] = sparse.load_npz(pattern.replace('<chrom>', chrom))\
                .tocsr()[row, col]

        print('  loading balanced data')
        balanced = np.zeros_like(raw, dtype=float)
        for r in range(self.design.shape[0]):
            balanced[:, r] = raw[:, r] / (bias[row, r] * bias[col, r])

        print('  scaling using size factors')
        size_factors = np.median(balanced / gmean(balanced, axis=1)[:, None],
                                 axis=0)
        scaled = balanced / size_factors
        del balanced

        print('  estimating dispersions')
        mean = np.dot(scaled, self.design) / np.sum(self.design, axis=0).values
        mean_wide = np.dot(mean, self.design.T)
        var = np.dot((scaled - mean_wide) ** 2, self.design) / \
            (np.sum(self.design, axis=0).values - 1)
        disp_idx = np.all((mean > self.disp_thresh_mean) & (var > 0), axis=1)
        disp = np.mean((var[disp_idx] - mean[disp_idx]) / mean[disp_idx] ** 2,
                       axis=0)
        disp = np.dot(disp, self.design.T)

        print('  fitting LRT models')
        # precompute product of all factors
        f = bias[row][disp_idx] * bias[col][disp_idx] * size_factors

        # fit LRT mu hat
        mu_hat_null = fit_mu_hat(raw[disp_idx, :], f, disp)
        mu_hat_alt = np.array(
            [fit_mu_hat(raw[disp_idx, :][:, self.design.values[:, c]],
                        f[:, self.design.values[:, c]],
                        disp[self.design.values[:, c]])
             for c in range(self.design.shape[1])]).T
        mu_hat_alt = np.dot(mu_hat_alt, self.design.T)

        print('  computing LRT results')
        null_ll = np.sum(logpmf(
            raw[disp_idx, :], mu_hat_null[:, None] * f, disp), axis=1)
        alt_ll = np.sum(logpmf(raw[disp_idx, :], mu_hat_alt * f, disp), axis=1)
        llr = null_ll - alt_ll
        lrt = np.exp(llr)
        pvalues = stats.chi2(self.design.shape[1] - 1).sf(-2 * np.log(lrt))

        if self.loop_patterns:
            print('  making loop_idx')
            loop_pixels = set.union(
                *sum((load_clusters(pattern.replace('<chrom>', chrom))
                      for pattern in self.loop_patterns), []))
            loop_idx = np.array([True if pixel in loop_pixels else False
                                 for pixel in zip(row[disp_idx],
                                                  col[disp_idx])])
            np.save('%s/loop_idx_%s.npy' % (self.outdir, chrom), loop_idx)

        print('  saving results to disk')
        np.save('%s/row_%s.npy' % (self.outdir, chrom), row)
        np.save('%s/col_%s.npy' % (self.outdir, chrom), col)
        np.save('%s/disp_idx_%s.npy' % (self.outdir, chrom), disp_idx)
        np.save('%s/raw_%s.npy' % (self.outdir, chrom), raw)
        np.save('%s/scaled_%s.npy' % (self.outdir, chrom), scaled)
        np.save('%s/mu_hat_null_%s.npy' % (self.outdir, chrom), mu_hat_null)
        np.save('%s/mu_hat_alt_%s.npy' % (self.outdir, chrom), mu_hat_alt)
        np.save('%s/llr_%s.npy' % (self.outdir, chrom), llr)
        np.save('%s/pvalues_%s.npy' % (self.outdir, chrom), pvalues)

    def process_all(self):
        for chrom in self.chroms:
            self.process_chrom(chrom)

    def bh(self):
        if self.loop_patterns:
            pvalues = [np.load('%s/pvalues_%s.npy' % (self.outdir, chrom))[
                           np.load('%s/loop_idx_%s.npy' % (self.outdir, chrom))]
                       for chrom in self.chroms]
        else:
            pvalues = [np.load('%s/pvalues_%s.npy' % (self.outdir, chrom))
                       for chrom in self.chroms]
        all_qvalues = adjust_pvalues(np.concatenate(pvalues))
        offset = 0
        for i, chrom in enumerate(self.chroms):
            np.save('%s/qvalues_%s.npy' % (self.outdir, chrom),
                    all_qvalues[offset:offset+len(pvalues[i])])
            offset += len(pvalues[i])

    def threshold_chrom(self, chrom, fdr=0.05, cluster_size=4):
        print('thresholding and clustering chrom %s' % chrom)
        # load everything
        row = np.load('%s/row_%s.npy' % (self.outdir, chrom))
        col = np.load('%s/col_%s.npy' % (self.outdir, chrom))
        disp_idx = np.load('%s/disp_idx_%s.npy' % (self.outdir, chrom))
        loop_idx = np.load('%s/loop_idx_%s.npy' % (self.outdir, chrom)) \
            if self.loop_patterns else np.ones(disp_idx.sum(), dtype=bool)
        qvalues = np.load('%s/qvalues_%s.npy' % (self.outdir, chrom))

        # upgrade fdr and cluster_size to list
        if not hasattr(fdr, '__len__'):
            fdr = [fdr]
        if not hasattr(cluster_size, '__len__'):
            cluster_size = [cluster_size]

        for f in fdr:
            # threshold on FDR
            sig_idx = qvalues < f
            insig_idx = qvalues >= f

            # gather and cluster sig and insig points
            sig_points = sparse.coo_matrix(
                (np.ones(sig_idx.sum(), dtype=bool),
                 (row[disp_idx][loop_idx][sig_idx],
                  col[disp_idx][loop_idx][sig_idx])))
            insig_points = sparse.coo_matrix(
                (np.ones(insig_idx.sum().sum(), dtype=bool),
                 (row[disp_idx][loop_idx][insig_idx],
                  col[disp_idx][loop_idx][insig_idx])))
            for s in cluster_size:
                sig_clusters = [c for c in find_clusters(sig_points)
                                if len(c) > s]
                insig_clusters = [c for c in find_clusters(insig_points)
                                  if len(c) > s]

                # save to disk
                save_clusters(sig_clusters, '%s/sig_%s_%g_%i.json' %
                              (self.outdir, chrom, f, s))
                save_clusters(insig_clusters, '%s/insig_%s_%g_%i.json' %
                              (self.outdir, chrom, f, s))

    def threshold_all(self, fdr=0.05, cluster_size=4):
        for chrom in self.chroms:
            self.threshold_chrom(chrom, fdr=fdr, cluster_size=cluster_size)
