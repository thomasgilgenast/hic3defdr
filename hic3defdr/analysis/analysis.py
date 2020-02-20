import os

import numpy as np
import scipy.sparse as sparse
import pandas as pd

from lib5c.util.statistics import adjust_pvalues

import hic3defdr.util.scaling as scaling
import hic3defdr.util.dispersion as dispersion
from hic3defdr.util.printing import eprint
from hic3defdr.util.matrices import sparse_union
from hic3defdr.util.clusters import load_clusters, save_clusters
from hic3defdr.util.cluster_table import clusters_to_table, \
    load_cluster_table, sort_cluster_table
from hic3defdr.util.lowess import lowess_fit, weighted_lowess_fit
from hic3defdr.util.lrt import lrt
from hic3defdr.util.thresholding import threshold_and_cluster, size_filter
from hic3defdr.util.classification import classify
from hic3defdr.util.progress import tqdm_maybe as tqdm
from hic3defdr.util.parallelization import parallel_apply, parallel_map


class AnalyzingHiC3DeFDR(object):
    """
    Mixin class containing analysis functions for HiC3DeFDR.
    """
    def prepare_data(self, chrom=None, norm='conditional_mor', n_bins=-1,
                     n_threads=-1, verbose=True):
        """
        Prepares raw and normalized data for analysis.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to prepare raw data for. Pass None to run
            for all chromosomes in series.
        norm : str
            The method to use to account for differences in sequencing depth.
            Valid options are:
                * simple_scaling: scale each replicate to equal total depth
                * median_of_ratios: use median of ratios normalization, ignoring
                  pixels at which any replicate has a zero
                * conditional_scaling: apply simple scaling independently at
                  each distance scale
                * conditional_mor: apply median of ratios independently at each
                  distance scale
        n_bins : int, optional
            Number of distance bins to use during scaling normalization if
            ``norm`` is one of the conditional options. Pass 0 or None to match
            pixels by exact distance. Pass -1 to use a reasonable default: 1/5
            of ``self.dist_thesh_max``.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        verbose : bool
            Pass False to silence reporting of progress to stderr.
        """
        if n_bins == -1:
            n_bins = self.dist_thresh_max / 5

        if chrom is None:
            if n_threads:
                parallel_apply(
                    self.prepare_data,
                    [{'chrom': c, 'norm': norm, 'n_bins': n_bins,
                      'verbose': False}
                     for c in self.chroms],
                    n_threads=n_threads
                )
            else:
                for chrom in self.chroms:
                    self.prepare_data(chrom=chrom, norm=norm, n_bins=n_bins)
            return
        eprint('preparing data for chrom %s' % chrom)
        eprint('  loading bias', skip=not verbose)
        bias = self.load_bias(chrom)

        eprint('  computing union pixel set', skip=not verbose)
        row, col = sparse_union(
            [pattern.replace('<chrom>', chrom)
             for pattern in self.raw_npz_patterns],
            dist_thresh=self.dist_thresh_max,
            bias=bias
        )

        eprint('  loading raw data', skip=not verbose)
        raw = np.zeros((len(row), len(self.raw_npz_patterns)), dtype=int)
        for i, pattern in enumerate(self.raw_npz_patterns):
            raw[:, i] = sparse.load_npz(pattern.replace('<chrom>', chrom)) \
                .tocsr()[row, col]

        eprint('  loading balanced data', skip=not verbose)
        balanced = np.zeros((len(row), len(self.raw_npz_patterns)), dtype=float)
        for r, pattern in enumerate(self.raw_npz_patterns):
            balanced[:, r] = sparse.load_npz(pattern.replace('<chrom>', chrom))\
                .tocsr()[row, col] / (bias[row, r] * bias[col, r])

        eprint('  computing size factors', skip=not verbose)
        if 'conditional' in norm:
            size_factors = scaling.__dict__[norm](balanced, col - row,
                                                  n_bins=n_bins)
        else:
            size_factors = scaling.__dict__[norm](balanced)
        scaled = balanced / size_factors

        eprint('  computing disp_idx', skip=not verbose)
        dist = col - row
        mean = np.dot(scaled, self.design) / np.sum(self.design, axis=0).values
        disp_idx = np.all(mean >= self.mean_thresh, axis=1) & \
            (dist >= self.dist_thresh_min)

        if self.loop_patterns:
            eprint('  making loop_idx', skip=not verbose)
            loop_pixels = set().union(
                *sum((load_clusters(pattern.replace('<chrom>', chrom))
                      for pattern in self.loop_patterns.values()), []))
            loop_idx = np.array([True if pixel in loop_pixels else False
                                 for pixel in zip(row[disp_idx],
                                                  col[disp_idx])])
            self.save_data(loop_idx, 'loop_idx', chrom)

        eprint('  saving data to disk', skip=not verbose)
        self.save_data(row, 'row', chrom)
        self.save_data(col, 'col', chrom)
        self.save_data(raw, 'raw', chrom)
        self.save_data(size_factors, 'size_factors', chrom)
        self.save_data(scaled, 'scaled', chrom)
        self.save_data(disp_idx, 'disp_idx', chrom)

    def estimate_disp(self, estimator='qcml', frac=None, auto_frac_factor=15.,
                      weighted_lowess=True, n_threads=-1):
        """
        Estimates dispersion parameters.

        Parameters
        ----------
        estimator : 'cml', 'qcml', 'mme', or a function
            Pass 'cml', 'qcml', 'mme' to use conditional maximum likelihood
            (CML), quantile-adjusted CML (qCML), or method of moments estimation
            (MME) to estimate the dispersion within each bin. Pass a function
            that takes in a (pixels, replicates) shaped array of data and
            returns a dispersion value to use that instead.
        frac : float, optional
            The lowess smoothing fraction to use when fitting the distance vs
            dispersion trend. Pass None to choose a value automatically.
        auto_frac_factor : float
            When ``frac`` is None, this factor scales the automatically
            determined fraction parameter.
        weighted_lowess : bool
            Whether or not to use a weighted lowess fit when fitting the
            smoothed dispersion curve.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple distance scales in parallel. Pass -1 to use
            as many threads as there are CPUs. Pass 0 to process the distance
            scales serially.
        """
        eprint('estimating dispersion')
        estimator = dispersion.__dict__[estimator] \
            if estimator in dispersion.__dict__ else estimator
        lowess_fn = weighted_lowess_fit if weighted_lowess else lowess_fit

        eprint('  loading data')
        disp_idx, disp_idx_offsets = self.load_data('disp_idx', 'all')
        row, offsets = self.load_data('row', 'all', idx=disp_idx)
        col, _ = self.load_data('col', 'all', idx=disp_idx)
        raw, _ = self.load_data('raw', 'all', idx=disp_idx)
        dist = col - row
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

        def _estimate(raw_slice, f_slice):
            return np.nan if not raw_slice.size \
                else estimator(raw_slice, f=f_slice)

        disp_per_dist = np.zeros((self.dist_thresh_max+1, self.design.shape[1]))
        disp = np.zeros((disp_idx.sum(), self.design.shape[1]))
        for c, cond in enumerate(self.design.columns):
            eprint('  estimating dispersion for condition %s' % cond)
            if n_threads:
                disp_per_dist[:, c] = parallel_map(
                    _estimate,
                    [{'raw_slice': raw[dist == d, :][:, self.design[cond]],
                      'f_slice': f[dist == d, :][:, self.design[cond]]}
                     for d in range(self.dist_thresh_max+1)],
                    n_threads=n_threads
                )
            else:
                for d in tqdm(range(self.dist_thresh_max+1)):
                    raw_slice = raw[dist == d, :][:, self.design[cond]]
                    f_slice = f[dist == d, :][:, self.design[cond]]
                    disp_per_dist[d, c] = np.nan if not raw_slice.size \
                        else estimator(raw_slice, f=f_slice)

            eprint('  fitting distance vs dispersion relationship')
            idx = np.isfinite(disp_per_dist[:, c])
            x = np.arange(self.dist_thresh_max+1)[idx]
            y = disp_per_dist[:, c][idx]
            lowess_kwargs = {'left_boundary': y[0]}
            if frac is not None:
                lowess_kwargs['frac'] = frac
            if weighted_lowess:
                lowess_kwargs['auto_frac_factor'] = auto_frac_factor
            disp_fn = lowess_fn(x, y, **lowess_kwargs)
            disp[:, c] = disp_fn(dist)
            self.save_disp_fn(cond, disp_fn)

        eprint('  saving estimated dispersions to disk')
        self.save_data(disp, 'disp', offsets)
        self.save_data(disp_per_dist, 'disp_per_dist')

    def lrt(self, chrom=None, refit_mu=True, n_threads=-1, verbose=True):
        """
        Runs the likelihood ratio test to test for differential interactions.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to run the LRT for. Pass None to run for
            all chromosomes in series.
        refit_mu : bool
            Pass True to refit the mean parameters in the NB models being
            compared in the LRT. Pass False to use the means across replicates
            directly, which is simpler and slightly faster but technically
            violates the assumptions of the LRT.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        verbose : bool
            Pass False to silence reporting of progress to stderr.
        """
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
        # we load size_factors the slow way because we don't know in advance if
        # it's 2 dimensional or not
        size_factors = self.load_data('size_factors', chrom)
        disp_idx = self.load_data('disp_idx', chrom)
        row = self.load_data('row', chrom, idx=disp_idx)
        col = self.load_data('col', chrom, idx=disp_idx)
        raw = self.load_data('raw', chrom, idx=disp_idx)
        disp = self.load_data('disp', chrom)

        eprint('  computing LRT results', skip=not verbose)
        if len(size_factors.shape) == 2:
            f = bias[row] * bias[col] * size_factors[disp_idx, :]
        else:
            f = bias[row] * bias[col] * size_factors
        pvalues, llr, mu_hat_null, mu_hat_alt = lrt(
            raw, f, np.dot(disp, self.design.values.T),
            self.design.values, refit_mu=refit_mu)

        eprint('  saving results to disk', skip=not verbose)
        self.save_data(pvalues, 'pvalues', chrom)
        self.save_data(llr, 'llr', chrom)
        self.save_data(mu_hat_null, 'mu_hat_null', chrom)
        self.save_data(mu_hat_alt, 'mu_hat_alt', chrom)

    def bh(self):
        """
        Applies BH-FDR control to p-values across all chromosomes to obtain
        q-values.

        Should only be run after all chromosomes have been processed through
        p-values.
        """
        eprint('applying BH-FDR correction')
        if self.loop_patterns:
            loop_idx, _ = self.load_data('loop_idx', 'all')
        else:
            loop_idx = None
        pvalues, offsets = self.load_data('pvalues', 'all', idx=loop_idx)
        all_qvalues = adjust_pvalues(pvalues)
        for i, chrom in enumerate(self.chroms):
            self.save_data(all_qvalues[offsets[i]:offsets[i+1]], 'qvalues',
                           chrom)

    def run_to_qvalues(self, norm='conditional_mor', n_bins_norm=-1,
                       estimator='qcml', frac=None, auto_frac_factor=15.,
                       weighted_lowess=True, refit_mu=True, n_threads=-1,
                       verbose=True):
        """
        Shortcut method to run the analysis to q-values.

        Parameters
        ----------
        norm : str
            The method to use to account for differences in sequencing depth.
            Valid options are:
                * simple_scaling: scale each replicate to equal total depth
                * median_of_ratios: use median of ratios normalization, ignoring
                  pixels at which any replicate has a zero
                * conditional_scaling: apply simple scaling independently at
                  each distance scale
                * conditional_mor: apply median of ratios independently at each
                  distance scale
        n_bins_norm : int, optional
            Number of distance bins to use during scaling normalization if
            ``norm`` is one of the conditional options. Pass 0 or None to match
            pixels by exact distance. Pass -1 to use a reasonable default: 1/5
            of ``self.dist_thesh_max``.
        estimator : 'cml', 'qcml', 'mme', or a function
            Pass 'cml', 'qcml', 'mme' to use conditional maximum likelihood
            (CML), qnorm-CML (qCML), or method of moments estimation (MME) to
            estimate the dispersion within each bin. Pass a function that takes
            in a (pixels, replicates) shaped array of data and returns a
            dispersion value to use that instead.
        frac : float, optional
            The lowess smoothing fraction to use when fitting the distance vs
            dispersion trend. Pass None to choose a value automatically.
        auto_frac_factor : float
            When ``frac`` is None, this factor scales the automatically
            determined fraction parameter.
        weighted_lowess : bool
            Whether or not to use a weighted lowess fit when fitting the
            smoothed dispersion curve.
        refit_mu : bool
            Pass True to refit the mean parameters in the NB models being
            compared in the LRT. Pass False to use the means across replicates
            directly, which is simpler and slightly faster but technically
            violates the assumptions of the LRT.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        verbose : bool
            Pass False to silence reporting of progress to stderr.
        """
        self.prepare_data(norm=norm, n_bins=n_bins_norm, n_threads=n_threads)
        self.estimate_disp(
            estimator=estimator, frac=frac, auto_frac_factor=auto_frac_factor,
            weighted_lowess=weighted_lowess, n_threads=n_threads)
        self.lrt(refit_mu=refit_mu, n_threads=n_threads)
        self.bh()

    def threshold(self, chrom=None, fdr=0.05, cluster_size=3, n_threads=-1):
        """
        Thresholds and clusters significantly differential pixels.

        Should only be run after q-values have been obtained.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to threshold. Pass None to threshold all
            chromosomes in series.
        fdr : float or list of float
            The FDR to threshold on. Pass a list to do a sweep in series.
        cluster_size : int or list of int
            Clusters smaller than this size will be filtered out. Pass a list to
            do a sweep in series.,
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        """
        if chrom is None:
            if n_threads:
                parallel_apply(
                    self.threshold,
                    [{'chrom': c, 'fdr': fdr, 'cluster_size': cluster_size}
                     for c in self.chroms],
                    n_threads=n_threads
                )
            else:
                for chrom in self.chroms:
                    self.threshold(chrom=chrom, fdr=fdr,
                                   cluster_size=cluster_size)
            return
        eprint('thresholding and clustering chrom %s' % chrom)
        # load everything
        row, col, qvalues = self.load_data('qvalues', chrom, coo=True)

        # upgrade fdr and cluster_size to list
        if not hasattr(fdr, '__len__'):
            fdr = [fdr]
        if not hasattr(cluster_size, '__len__'):
            cluster_size = [cluster_size]

        for f in fdr:
            sig_clusters, insig_clusters = threshold_and_cluster(
                qvalues, row, col, fdr)
            for s in cluster_size:
                # threshold on cluster size
                filtered_sig_clusters = size_filter(sig_clusters, s)
                filtered_insig_clusters = size_filter(insig_clusters, s)
                # save to disk
                sig_outfile = '%s/sig_%g_%i_%s.json' % \
                    (self.outdir, f, s, chrom)
                insig_outfile = '%s/insig_%g_%i_%s.json' % \
                    (self.outdir, f, s, chrom)
                save_clusters(filtered_sig_clusters, sig_outfile)
                save_clusters(filtered_insig_clusters, insig_outfile)
                if self.res is not None:
                    clusters_to_table(filtered_sig_clusters, chrom, self.res)\
                        .to_csv(sig_outfile.replace('.json', '.tsv'), sep='\t')
                    clusters_to_table(filtered_insig_clusters, chrom, self.res)\
                        .to_csv(insig_outfile.replace('.json', '.tsv'),
                                sep='\t')

    def classify(self, chrom=None, fdr=0.05, cluster_size=3, n_threads=-1):
        """
        Classifies significantly differential pixels according to which
        condition they are strongest in.

        Parameters
        ----------
        chrom : str
            The chromosome to classify significantly differential pixels on.
            Pass None to run for all chromosomes in series.
        fdr : float or list of float
            The FDR threshold used to identify clusters of significantly
            differential pixels via ``self.threshold()``. Pass a list to do a
            sweep in series.
        cluster_size : int or list of int
            The cluster size threshold used to identify clusters of
            significantly differential pixels via ``threshold()``. Pass a list
            to do a sweep in series.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        """
        if chrom is None:
            if n_threads:
                parallel_apply(
                    self.classify,
                    [{'chrom': c, 'fdr': fdr, 'cluster_size': cluster_size}
                     for c in self.chroms],
                    n_threads=n_threads
                )
            else:
                for chrom in self.chroms:
                    self.classify(chrom=chrom, fdr=fdr,
                                  cluster_size=cluster_size)
            return
        eprint('classifying differential interactions on chrom %s' % chrom)
        # load everything
        disp_idx = self.load_data('disp_idx', chrom)
        loop_idx = self.load_data('loop_idx', chrom)
        row = self.load_data('row', chrom, idx=(disp_idx, loop_idx))
        col = self.load_data('col', chrom, idx=(disp_idx, loop_idx))
        mu_hat_alt = self.load_data('mu_hat_alt', chrom, idx=loop_idx)

        # upgrade fdr and cluster_size to list
        if not hasattr(fdr, '__len__'):
            fdr = [fdr]
        if not hasattr(cluster_size, '__len__'):
            cluster_size = [cluster_size]

        for f in fdr:
            for s in cluster_size:
                infile = '%s/sig_%g_%i_%s.json' % (self.outdir, f, s, chrom)
                if not os.path.isfile(infile):
                    self.threshold(chrom=chrom, fdr=f, cluster_size=s)
                sig_clusters = load_clusters(infile)
                class_clusters = classify(row, col, mu_hat_alt, sig_clusters)
                for i, c in enumerate(class_clusters):
                    outfile = '%s/%s_%g_%i_%s.json' % \
                        (self.outdir, self.design.columns[i], f, s, chrom)
                    save_clusters(c, outfile)
                    if self.res is not None:
                        clusters_to_table(c, chrom, self.res)\
                            .to_csv(outfile.replace('.json', '.tsv'), sep='\t')

    def collect(self, fdr=0.05, cluster_size=3, n_threads=-1):
        """
        Collects information on thresholded and classified differential
        interactions into a single TSV output file.

        Parameters
        ----------
        fdr : float or list of float
            The FDR threshold used to identify clusters of significantly
            differential pixels via ``self.threshold()``. Pass a list to do a
            sweep in series.
        cluster_size : int or list of int
            The cluster size threshold used to identify clusters of
            significantly differential pixels via ``threshold()``. Pass a list
            to do a sweep in series.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        """
        if self.res is None:
            raise ValueError(
                'the collect() step can only be run if the res kwarg was '
                'passed during construction of the HiC3DeFDR object; please '
                'run the classify() step instead or re-create the HiC3DeFDR '
                'object (you do not need to re-run any other steps)'
            )

        eprint('collecting differential interactions')

        # upgrade fdr and cluster_size to list
        if not hasattr(fdr, '__len__'):
            fdr = [fdr]
        if not hasattr(cluster_size, '__len__'):
            cluster_size = [cluster_size]

        for f in fdr:
            for s in cluster_size:
                # file pattern where we expect tables to be present
                pattern = '%s/<class>_%g_%i_<chrom>.tsv' % (self.outdir, f, s)

                # ensure that all outputs are present
                if not all(os.path.isfile(pattern
                                          .replace('<class>', 'insig')
                                          .replace('<chrom>', chrom))
                           for chrom in self.chroms):
                    self.threshold(fdr=f, cluster_size=s, n_threads=n_threads)
                if not all(os.path.isfile(pattern
                                          .replace('<class>', c)
                                          .replace('<chrom>', chrom))
                           for c in self.design.columns
                           for chrom in self.chroms):
                    self.classify(fdr=f, cluster_size=s, n_threads=n_threads)

                # output file we want to write the final table to
                outfile = '%s/results_%g_%i.tsv' % (self.outdir, f, s)

                # collect cluster tables
                tables = []
                for chrom in self.chroms:
                    df = load_cluster_table(pattern
                                            .replace('<class>', 'insig')
                                            .replace('<chrom>', chrom))
                    df['classification'] = 'constitutive'
                    tables.append(df)
                    for c in self.design.columns:
                        df = load_cluster_table(pattern
                                                .replace('<class>', c)
                                                .replace('<chrom>', chrom))
                        df['classification'] = c
                        tables.append(df)

                # write to disk
                sort_cluster_table(pd.concat(tables)).to_csv(outfile, sep='\t')
