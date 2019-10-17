import os

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import dill as pickle

from lib5c.util.system import check_outdir
from lib5c.util.statistics import adjust_pvalues

import hic3defdr.scaling as scaling
import hic3defdr.dispersion as dispersion
from hic3defdr.logging import eprint
from hic3defdr.matrices import sparse_union
from hic3defdr.clusters import load_clusters, save_clusters
from hic3defdr.lowess import lowess_fit
from hic3defdr.lrt import lrt
from hic3defdr.thresholding import threshold_and_cluster, size_filter
from hic3defdr.classification import classify
from hic3defdr.simulation import simulate
from hic3defdr.evaluation import make_y_true, evaluate
from hic3defdr.plotting.distance_dependence import plot_dd_curves
from hic3defdr.plotting.histograms import plot_pvalue_histogram
from hic3defdr.plotting.dispersion import plot_mvr
from hic3defdr.plotting.ma import plot_ma
from hic3defdr.plotting.grid import plot_grid
from hic3defdr.progress import tqdm_maybe as tqdm
from hic3defdr.parallelization import parallel


class HiC3DeFDR(object):
    """
    Main object for hic3defdr analysis.

    Attributes
    ----------
    raw_npz_patterns : list of str
        File path patterns to ``scipy.sparse`` formatted NPZ files containing
        raw contact matrices for each replicate, in order. Each file path
        pattern should contain at least one '<chrom>' which will be replaced
        with the chromosome name when loading data for specific chromosomes.
    bias_patterns : list of str
        File path patterns to ``np.savetxt()`` formatted files containing bias
        vector information for each replicate, in order. ach file path pattern
        should contain at least one '<chrom>' which will be replaced with the
        chromosome name when loading data for specific chromosomes.
    chroms : list of str
        List of chromosome names as strings. These names will be substituted in
        for '<chroms>' in the ``raw_npz_patterns`` and ``bias_patterns``.
    design : pd.DataFrame or str
        Pass a DataFrame with boolean dtype whose rows correspond to replicates
        and whose columns correspond to conditions. Replicate and condition
        names will be inferred from the row and column labels, respectively. If
        you pass a string, the DataFrame will be loaded via
        ``pd.read_csv(design, index_col=0)``.
    outdir : str
        Specify a directory to store the results of the analysis. Two different
        HiC3DeFDR analyses cannot co-exist in the same directory. The directory
        will be created if it does not exist.
    dist_thresh_min, dist_thresh_max : int
        The minimum and maximum interaction distance (in bin units) to include
        in the analysis.
    bias_thresh : float
        Bins with a bias factor below this threshold or above its reciprocal in
        any replicate will be filtered out of the analysis.
    mean_thresh : float
        Pixels with mean value below this threshold will be filtered out at the
        dispersion fitting stage.
    loop_patterns : dict of str
        Keys should be condition names as strings, values should be file path
        patterns to sparse JSON formatted cluster files representing called
        loops in that condition. Each file path pattern should contain at least
        one '<chrom>' which will be replaced with the chromosome name when
        loading data for specific chromosomes.
    """

    def __init__(self, raw_npz_patterns, bias_patterns, chroms, design, outdir,
                 dist_thresh_min=4, dist_thresh_max=500, bias_thresh=0.1,
                 mean_thresh=5.0, loop_patterns=None):
        """
        Base constructor. See ``help(HiC3DeFDR)`` for details.
        """
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
        self.mean_thresh = mean_thresh
        self.loop_patterns = loop_patterns
        state = self.__dict__.copy()
        del state['outdir']
        check_outdir(self.picklefile)
        with open(self.picklefile, 'wb') as handle:
            pickle.dump(state, handle, -1)

    @property
    def picklefile(self):
        return '%s/pickle' % self.outdir

    @classmethod
    def load(cls, outdir):
        """
        Loads a HiC3DeFDR analysis object from disk.

        It is safe to have multiple instances of the same analysis open at once.

        Parameters
        ----------
        outdir : str
            Folder path to where the HiC3DeFDR was saved.

        Returns
        -------
        HiC3DeFDR
            The loaded object.
        """
        with open('%s/pickle' % outdir, 'rb') as handle:
            return cls(outdir=outdir, **pickle.load(handle))

    def load_bias(self, chrom):
        """
        Loads the bias matrix for one chromosome.

        The rows of the bias matrix correspond to bin indices along the
        chromosome. The columns correspond to the replicates.

        The bias factors for bins that fail ``bias_thresh`` are set to zero.
        This is designed so that all pixels in these bins get dropped during
        union pixel set computation.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to load the bias matrix for.

        Returns
        -------
        np.ndarray
            The bias matrix.
        """
        bias = np.array([np.loadtxt(pattern.replace('<chrom>', chrom))
                         for pattern in self.bias_patterns]).T
        bias[(np.any(bias < self.bias_thresh, axis=1)) |
             (np.any(bias > 1. / self.bias_thresh, axis=1)), :] = 0
        return bias

    def load_data(self, name, chrom=None, idx=None):
        """
        Loads arbitrary data for one chromosome or all chromosomes.

        Parameters
        ----------
        name : str
            The name of the data to load.
        chrom : str, optional
            The name of the chromosome to load data for. Pass None if this data
            is not organized by chromosome. Pass 'all' to load data for all
            chromosomes.
        idx : np.ndarray, optional
            Pass a boolean array to load only a subset of the data. Useful in
            combination with ``chrom='all'`` to reduce memory usage. Ignored if
            ``chrom`` is not 'all'.

        Returns
        -------
        data or (concatenated_data, offsets) : np.ndarray
            The loaded data for one chromosome, or a tuple of the concatenated
            data and an array of offsets per chromosome. ``offsets`` satisfies
            the following properties: ``offsets[0] == 0``,
            ``offsets[-1] == concatenated_data.shape[0]``, and
            ``concatenated_data[offsets[i]:offsets[i+1], :]`` contains the data
            for the ``i``th chromosome.
        """
        # tackle simple cases first
        if chrom is None:
            return np.load('%s/%s.npy' % (self.outdir, name))
        if chrom != 'all':
            return np.load('%s/%s_%s.npy' % (self.outdir, name, chrom))

        # idx is genome-wide, this tracks where we are in idx so that we can
        # find a subset of idx that aligns with the current chrom
        idx_offset = 0

        # list of data arrays per chromosome
        all_data = []

        # running total of the sizes of the elements of all data
        offset = 0

        # saves value of offset after each chrom
        offsets = [0]

        # loop over chroms
        for chrom in self.chroms:
            data = np.load('%s/%s_%s.npy' % (self.outdir, name, chrom))
            if idx is not None:
                full_data_size = data.shape[0]
                data = data[idx[idx_offset:idx_offset+full_data_size]]
                idx_offset += full_data_size
            offset += data.shape[0]
            offsets.append(offset)
            all_data.append(data)
        return np.concatenate(all_data), np.array(offsets)

    def save_data(self, data, name, chrom=None):
        """
        Saves arbitrary data for one chromosome to disk.

        Parameters
        ----------
        data : np.ndarray
            The data to save.
        name : str
            The name of the data to save.
        chrom : str or np.ndarray, optional
            The name of the chromosome to save data for, or None if this data is
            not organized by chromosome. Pass an np.ndarray of offsets to save
            data for all chromosomes.
        """
        if chrom is None:
            np.save('%s/%s.npy' % (self.outdir, name), data)
        elif isinstance(chrom, np.ndarray):
            for i, c in enumerate(self.chroms):
                self.save_data(data[chrom[i]:chrom[i + 1]], name, c)
        else:
            np.save('%s/%s_%s.npy' % (self.outdir, name, chrom), data)

    def load_disp_fn(self, cond):
        """
        Loads the fitted dispersion function for a specific condition from disk.

        Parameters
        ----------
        cond : str
            The condition to load the dispersion function for.

        Returns
        -------
        function
            Vectorized. Takes in the value of the covariate the dispersion was
            fitted against and returns the appropriate dispersion.
        """
        picklefile = '%s/disp_fn_%s.pickle' % (self.outdir, cond)
        with open(picklefile, 'rb') as handle:
            return pickle.load(handle)

    def save_disp_fn(self, cond, disp_fn):
        """
        Saves the fitted dispersion function for a specific condition and
        chromosome to disk.

        Parameters
        ----------
        cond : str
            The condition to save the dispersion function for.
        disp_fn : function
            The dispersion function to save.
        """
        picklefile = '%s/disp_fn_%s.pickle' % (self.outdir, cond)
        with open(picklefile, 'wb') as handle:
            return pickle.dump(disp_fn, handle, -1)

    def prepare_data(self, chrom=None, norm='conditional_mor', n_bins=100,
                     n_threads=0, verbose=True):
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
            pixels by exact distance.
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
                parallel(
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
        disp_idx = np.all(mean > self.mean_thresh, axis=1) & \
            (dist >= self.dist_thresh_min)

        eprint('  saving data to disk', skip=not verbose)
        self.save_data(row, 'row', chrom)
        self.save_data(col, 'col', chrom)
        self.save_data(raw, 'raw', chrom)
        self.save_data(size_factors, 'size_factors', chrom)
        self.save_data(scaled, 'scaled', chrom)
        self.save_data(disp_idx, 'disp_idx', chrom)

    def estimate_disp(self, estimator='cml'):
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
        """
        eprint('estimating dispersion')
        estimator = dispersion.__dict__[estimator] \
            if estimator in dispersion.__dict__ else estimator

        eprint('  loading data')
        disp_idx, _ = self.load_data('disp_idx', 'all')
        row, offsets = self.load_data('row', 'all', idx=disp_idx)
        col, _ = self.load_data('col', 'all', idx=disp_idx)
        scaled, _ = self.load_data('scaled', 'all', idx=disp_idx)
        dist = col - row

        disp_per_dist = np.zeros((self.dist_thresh_max+1, self.design.shape[1]))
        disp = np.zeros((disp_idx.sum(), self.design.shape[1]))
        for c, cond in enumerate(self.design.columns):
            eprint('  estimating dispersion for condition %s' % cond)
            for d in tqdm(range(self.dist_thresh_max+1)):
                data = scaled[dist == d, :][:, self.design[cond]]
                disp_per_dist[d, c] = np.nan if not data.size \
                    else estimator(data)
            idx = np.isfinite(disp_per_dist[:, c])
            disp_fn = lowess_fit(np.arange(self.dist_thresh_max+1)[idx],
                                 disp_per_dist[:, c][idx], logx=True, logy=True)
            disp[:, c] = disp_fn(dist)
            self.save_disp_fn(cond, disp_fn)

        eprint('  saving estimated dispersions to disk')
        self.save_data(disp, 'disp', offsets)
        self.save_data(disp_per_dist, 'disp_per_dist')

    def lrt(self, chrom=None, refit_mu=True, n_threads=0, verbose=True):
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
                parallel(
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
        disp = self.load_data('disp', chrom)

        eprint('  computing LRT results', skip=not verbose)
        if len(size_factors.shape) == 2:
            f = bias[row][disp_idx] * bias[col][disp_idx] * \
                size_factors[disp_idx, :]
        else:
            f = bias[row][disp_idx] * bias[col][disp_idx] * size_factors
        pvalues, llr, mu_hat_null, mu_hat_alt = lrt(
            raw[disp_idx, :], f, np.dot(disp, self.design.values.T),
            self.design.values, refit_mu=refit_mu)

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

    def bh(self):
        """
        Applies BH-FDR control to p-values across all chromosomes to obtain
        q-values.

        Should only be run after all chromosomes have been processed through
        p-values.
        """
        eprint('applying BH-FDR correction')
        if self.loop_patterns:
            pvalues = [
                self.load_data('pvalues', chrom)
                [self.load_data('loop_idx', chrom)]
                for chrom in self.chroms
            ]
        else:
            pvalues = [self.load_data('pvalues', chrom)
                       for chrom in self.chroms]
        all_qvalues = adjust_pvalues(np.concatenate(pvalues))
        offset = 0
        for i, chrom in enumerate(self.chroms):
            self.save_data(all_qvalues[offset:offset + len(pvalues[i])],
                           'qvalues', chrom)
            offset += len(pvalues[i])

    def run_to_qvalues(self, norm='conditional_mor', n_bins_norm=100,
                       estimator='cml', refit_mu=True, n_threads=0,
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
            pixels by exact distance.
        estimator : 'cml', 'qcml', 'mme', or a function
            Pass 'cml', 'qcml', 'mme' to use conditional maximum likelihood
            (CML), qnorm-CML (qCML), or method of moments estimation (MME) to
            estimate the dispersion within each bin. Pass a function that takes
            in a (pixels, replicates) shaped array of data and returns a
            dispersion value to use that instead.
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
        self.estimate_disp(estimator=estimator)
        self.lrt(refit_mu=refit_mu, n_threads=n_threads)
        self.bh()

    def threshold(self, chrom=None, fdr=0.05, cluster_size=3, n_threads=0):
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
                parallel(
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
        disp_idx = self.load_data('disp_idx', chrom)
        loop_idx = self.load_data('loop_idx', chrom) \
            if self.loop_patterns else np.ones(disp_idx.sum(), dtype=bool)
        row = self.load_data('row', chrom)[disp_idx][loop_idx]
        col = self.load_data('col', chrom)[disp_idx][loop_idx]
        qvalues = self.load_data('qvalues', chrom)

        # upgrade fdr and cluster_size to list
        if not hasattr(fdr, '__len__'):
            fdr = [fdr]
        if not hasattr(cluster_size, '__len__'):
            cluster_size = [cluster_size]

        for f in fdr:
            sig_clusters, insig_clusters = threshold_and_cluster(
                qvalues, row, col, fdr)
            for s in cluster_size:
                # threshold on cluster size and save to disk
                save_clusters(size_filter(sig_clusters, s),
                              '%s/sig_%g_%i_%s.json' %
                              (self.outdir, f, s, chrom))
                save_clusters(size_filter(insig_clusters, s),
                              '%s/insig_%g_%i_%s.json' %
                              (self.outdir, f, s, chrom))

    def classify(self, chrom=None, fdr=0.05, cluster_size=3, n_threads=0):
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
            differential pixels via ``threshold_chrom()``. Pass a list to do a
            sweep in series.
        cluster_size : int or list of int
            The cluster size threshold used to identify clusters of
            significantly differential pixels via ``threshold_chrom()``. Pass a
            list to do a sweep in series.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        """
        if chrom is None:
            if n_threads:
                parallel(
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
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        mu_hat_alt = self.load_data('mu_hat_alt', chrom)
        disp_idx = self.load_data('disp_idx', chrom)
        loop_idx = self.load_data('loop_idx', chrom) \
            if self.loop_patterns else np.ones(disp_idx.sum(), dtype=bool)

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
                class_clusters = classify(
                    row[disp_idx][loop_idx],
                    col[disp_idx][loop_idx],
                    mu_hat_alt[loop_idx],
                    sig_clusters
                )
                for i, c in enumerate(class_clusters):
                    save_clusters(
                        c, '%s/%s_%g_%i_%s.json' %
                           (self.outdir, self.design.columns[i], f, s, chrom))

    def simulate(self, cond, chrom=None, beta=0.5, p_diff=0.4, scramble=False,
                 skip_bias=False, loop_pattern=None, outdir='sim', n_threads=0,
                 verbose=True):
        """
        Simulates raw contact matrices based on previously fitted scaled means
        and dispersions in a specific condition.

        This function will only work when the design has exactly two conditions
        and equal numbers of replicates per condition.

        Can only be run after ``estimate_dispersions()`` has been run.

        Parameters
        ----------
        cond : str
            Name of the condition to base the simulation on.
        chrom : str, optional
            Name of the chromosome to simulate. Pass None to simulate all
            chromosomes in series.
        beta : float
            The effect size of the loop perturbations to use when simulating.
            Perturbed loops will be strengthened or weakened by this fraction of
            their original strength.
        p_diff : float
            This fraction of loops will be perturbed across the simulated
            conditions. The remainder will be constitutive.
        scramble : bool
            Pass True to scramble the bias vectors and size factors when
            assigning them to simulated replicates.
        skip_bias : bool
            Pass True to set all bias factors and size factors to 1,
            effectively simulating "unbiased" raw data.
        loop_pattern : str, optional
            File path pattern to sparse JSON formatted cluster files
            representing loop cluster locations for the simulation. Should
            contain at least one '<chrom>' which will be replaced with the
            chromosome name when loading data for specific chromosomes. Pass
            None to use ``self.loop_patterns[cond]``.
        outdir : str
            Path to a directory to store the simulated data to.
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
                parallel(
                    self.simulate,
                    [{'cond': cond, 'chrom': c, 'beta': beta, 'p_diff': p_diff,
                      'loop_pattern': loop_pattern, 'outdir': outdir,
                      'verbose': False}
                     for c in self.chroms],
                    n_threads=n_threads
                )
            else:
                for chrom in self.chroms:
                    self.simulate(cond, chrom=chrom, beta=beta, p_diff=p_diff,
                                  loop_pattern=loop_pattern, outdir=outdir)
            return
        eprint('simulating data for chrom %s' % chrom)
        # resolve loop_pattern
        if loop_pattern is None:
            loop_pattern = self.loop_patterns[cond]

        # load everything
        bias = self.load_bias(chrom)
        size_factors = self.load_data('size_factors', chrom)
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        scaled = self.load_data('scaled', chrom)[:, self.design[cond]]
        disp_fn = self.load_disp_fn(cond)
        clusters = load_clusters(loop_pattern.replace('<chrom>', chrom))

        # compute pixel-wise mean of normalized data
        mean = np.mean(scaled, axis=1)

        # book keeping
        check_outdir('%s/' % outdir)
        n_sim = size_factors.shape[-1]
        repnames = sum((['%s%i' % (c, i+1) for i in range(n_sim/2)]
                        for c in ['A', 'B']), [])

        # write design to disk if not present
        design_file = '%s/design.csv' % outdir
        if not os.path.isfile(design_file):
            pd.DataFrame(
                {'A': [1]*(n_sim/2) + [0]*(n_sim/2),
                 'B': [0]*(n_sim/2) + [1]*(n_sim/2)},
                dtype=bool,
                index=repnames
            ).to_csv(design_file)

        # rewrite size_factor matrix in terms of distance
        if len(size_factors.shape) == 2:
            eprint('  converting size factors', skip=not verbose)
            dist = col - row
            n_dists = dist.max() + 1
            new_size_factors = np.zeros((n_dists, size_factors.shape[1]))
            for d in tqdm(range(n_dists)):
                idx = np.argmax(dist == d)
                new_size_factors[d, :] = size_factors[idx, :]
            size_factors = new_size_factors

        # get rid of bias
        if skip_bias:
            bias = np.ones_like(bias)
            size_factors = np.ones_like(size_factors)

        # scramble bias and size_factors
        if scramble:
            bias = bias[:, (np.arange(n_sim)+1) % n_sim]
            if len(size_factors.shape) == 1:
                size_factors = size_factors[(np.arange(n_sim)+3) % n_sim]
            else:
                size_factors = size_factors[:, (np.arange(n_sim)+3) % n_sim]

        # simulate and save
        classes, sim_iter = simulate(
            row, col, mean, disp_fn, bias, size_factors, clusters, beta=beta,
            p_diff=p_diff, trend='dist', verbose=verbose)
        np.savetxt('%s/labels_%s.txt' % (outdir, chrom), classes, fmt='%s')
        for rep, csr in zip(repnames, sim_iter):
            sparse.save_npz('%s/%s_%s_raw.npz' % (outdir, rep, chrom), csr)

    def evaluate(self, cluster_pattern, label_pattern, outfile='eval.npz'):
        """
        Evaluates the results of this analysis, comparing it to true labels.

        Parameters
        ----------
        cluster_pattern : str
            File path pattern to sparse JSON formatted cluster files
            representing loop cluster locations. Should contain at least one
            '<chrom>' which will be replaced with the chromosome name when
            loading data for specific chromosomes. Pass a condition name to use
            ``self.loop_patterns[cluster_pattern]`` instead.
        label_pattern : str
            File path pattern to true label files for each chromosome. Should
            contain at least one '<chrom>' which will be replaced with the
            chromosome name when loading data for specific chromosomes. Files
            should be loadable with ``np.loadtxt(..., dtype='|S7')`` to yield a
            vector of true labels parallel to the clusters pointed to by
            ``cluster_pattern``.
        outfile : str
            Name of a file to save the evaluation results to inside this
            object's ``outdir``.
        """
        # resolve case where a condition name was passed to cluster_pattern
        if cluster_pattern in self.loop_patterns.keys():
            cluster_pattern = self.loop_patterns[cluster_pattern]

        # make y_true one chrom at a time
        y_true = []
        for chrom in self.chroms:
            disp_idx = self.load_data('disp_idx', chrom)
            loop_idx = self.load_data('loop_idx', chrom)
            row = self.load_data('row', chrom)[disp_idx][loop_idx]
            col = self.load_data('col', chrom)[disp_idx][loop_idx]
            clusters = load_clusters(cluster_pattern.replace('<chrom>', chrom))
            labels = np.loadtxt(label_pattern.replace('<chrom>', chrom),
                                dtype='|S7')
            y_true.append(make_y_true(row, col, clusters, labels))
        y_true = np.concatenate(y_true)

        # load qvalues
        qvalues = np.concatenate([self.load_data('qvalues', chrom)
                                  for chrom in self.chroms])

        # evaluate and save to disk
        fdr, fpr, tpr, thresh = evaluate(y_true, qvalues)

        # save to disk
        np.savez('%s/%s' % (self.outdir, outfile),
                 **{'fdr': fdr, 'fpr': fpr, 'tpr': tpr, 'thresh': thresh})

    def plot_dd_curves(self, chrom, log=True, **kwargs):
        """
        Plots the distance dependence curve before and after size factor
        adjustment.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to plot the curve for.
        log : bool
            Whether or not to log the axes of the plot.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # load everything
        bias = self.load_bias(chrom)
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        raw = self.load_data('raw', chrom)
        scaled = self.load_data('scaled', chrom)

        # compute balanced
        balanced = np.zeros_like(raw, dtype=float)
        for r in range(self.design.shape[0]):
            balanced[:, r] = raw[:, r] / (bias[row, r] * bias[col, r])

        return plot_dd_curves(row, col, balanced, scaled, self.design, log=log,
                              **kwargs)

    def plot_dispersion_fit(self, cond, xaxis='dist', yaxis='disp',
                            dist_max=200, scatter_fit=-1, scatter_size=36,
                            distance=None, hexbin=False, logx=False, logy=False,
                            **kwargs):
        """
        Plots a hexbin plot of pixel-wise distance vs either dispersion or
        variance, overlaying the estimated and fitted dispersions.

        Parameters
        ----------
        cond : str
            The name of the chromosome and condition, respectively, to plot the
            fit for.
        xaxis : 'mean' or 'dist'
            What to plot on the x-axis.
        yaxis : 'disp' or 'var'
            What to plot on the y-axis.
        dist_max : int
            If ``xaxis`` is 'dist', the maximum distance to include on the plot
            in bin units.
        scatter_fit : int
            Pass a nonzero integer to draw the fitted dispersions passed in
            ``disp`` as a scatterplot of ``scatter_fit`` selected points. Pass
            -1 to plot the fitted dispersions passed in ``disp`` as a curve.
            Pass 0 to omit plotting the dispersion estimates altogether.
        scatter_size : int
            The marker size when plotting scatterplots.
        distance : int, optional
            Pick a specific distance in bin units to plot only interactions at
            that distance.
        hexbin : bool
            Pass False to skip plotting the hexbin plot, leaving only the
            estimated variances or dispersions.
        logx, logy : bool
            Whether or not to log the x- or y-axis, respectively.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # identify cond_idx
        cond_idx = self.design.columns.tolist().index(cond)

        # load everything
        disp_idx, _ = self.load_data('disp_idx', 'all')
        scaled = self.load_data(
            'scaled', 'all', idx=disp_idx)[0][:, self.design[cond]]
        disp = self.load_data('disp', 'all')[0][:, cond_idx]
        try:
            disp_per_dist = self.load_data('disp_per_dist')[:, cond_idx]
            idx = np.isfinite(disp_per_dist)
            disp_per_bin = disp_per_dist[idx]
            dist_per_bin = np.arange(self.dist_thresh_max + 1)[idx]
        except IOError:
            disp_per_bin = None
            dist_per_bin = None
        row, _ = self.load_data('row', 'all', idx=disp_idx)
        col, _ = self.load_data('col', 'all', idx=disp_idx)
        dist = col - row

        # compute mean and sample variance
        mean = np.mean(scaled, axis=1)
        var = np.var(scaled, ddof=1, axis=1)

        # resolve distance
        if distance is not None:
            dist_idx = dist == distance
            mean = mean[dist_idx]
            var = var[dist_idx]
            dist = None
            disp = np.ones(dist_idx.sum()) * disp_per_dist[distance]
            dist_per_bin = None
            disp_per_bin = None
            fit_align_dist = False
        else:
            fit_align_dist = xaxis == 'mean' or yaxis == 'var'

        return plot_mvr(
            pixel_mean=mean,
            pixel_var=var,
            pixel_dist=dist,
            pixel_disp_fit=disp,
            dist_per_bin=dist_per_bin,
            disp_per_bin=disp_per_bin,
            fit_align_dist=fit_align_dist,
            xaxis=xaxis, yaxis=yaxis,
            dist_max=dist_max, mean_min=self.mean_thresh,
            scatter_fit=scatter_fit, scatter_size=scatter_size, hexbin=hexbin,
            logx=logx, logy=logy, **kwargs
        )

    def plot_pvalue_distribution(self, idx='disp', **kwargs):
        """
        Plots the p-value distribution across all chromosomes.

        Parameters
        ----------
        idx : {'disp', 'loop'}
            Pass 'disp' to plot p-values for all points for which dispersion was
            estimated. Pass 'loop' to plot p-values for all points which are in
            loops (available only if ``loop_patterns`` was passed to the
            constructor).
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # load everything
        if idx == 'loop':
            loop_idx, _ = self.load_data('loop_idx', 'all')
            pvalues, _ = self.load_data('pvalues', 'all', idx=loop_idx)
        elif idx == 'disp':
            pvalues, _ = self.load_data('pvalues', 'all')
        else:
            raise ValueError('idx must be loop or disp')

        # plot
        return plot_pvalue_histogram(pvalues, **kwargs)

    def plot_qvalue_distribution(self, **kwargs):
        """
        Plots the q-value distribution across all chromosomes.

        Parameters
        ----------
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # load everything
        qvalues, _ = self.load_data('qvalues', 'all')

        # plot
        return plot_pvalue_histogram(qvalues, xlabel='qvalue', **kwargs)

    def plot_ma(self, fdr=0.05, conds=None, include_non_loops=True, s=1,
                **kwargs):
        """
        Plots an MA plot for a given chromosome.

        Parameters
        ----------
        fdr : float
            The threshold to use for labeling significantly differential loop
            pixels.
        conds : tuple of str, optional
            Pass a tuple of two condition names to compare those two
            conditions. Pass None to compare the first two conditions.
        include_non_loops : bool
            Whether or not to include non-looping pixels in the MA plot.
        s : float
            The marker size to use for the scatterplot.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # resolve conds
        if conds is None:
            conds = self.design.columns.tolist()[:2]
        cond_idx = [self.design.columns.tolist().index(cond) for cond in conds]

        # load data
        disp_idx, _ = self.load_data('disp_idx', 'all')
        loop_idx, _ = self.load_data('loop_idx', 'all')
        scaled, _ = self.load_data('scaled', 'all', idx=disp_idx)
        qvalues, _ = self.load_data('qvalues', 'all')

        # compute mean
        mean = np.dot(scaled, self.design) / np.sum(self.design, axis=0).values
        mean = mean[:, cond_idx]

        # prepare sig_idx
        sig_idx = qvalues < fdr

        # plot
        if include_non_loops:
            plot_ma(mean, sig_idx, loop_idx=loop_idx, names=conds, s=s,
                    **kwargs)
        else:
            plot_ma(mean[loop_idx], sig_idx, names=conds, s=s, **kwargs)

    def plot_grid(self, chrom, i, j, w, vmax=100, fdr=0.05, cluster_size=3,
                  fdr_vmid=0.05,
                  color_cycle=('blue', 'green', 'purple', 'yellow', 'cyan',
                               'red'),
                  despine=False, **kwargs):
        """
        Plots a combination visualization grid focusing on a specific pixel on a
        specific chromosome, combining heatmaps, cluster outlines, and
        stripplots.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to slice matrices from.
        i, j : int
            The row and column index of the pixel to focus on.
        w : int
            The size of the heatmap will be ``2*w`` bins in each dimension.
        vmax : float
            The maximum of the colorscale to use when plotting normalized
            heatmaps.
        fdr : float
            The FDR threshold to use when outlining clusters.
        cluster_size : int
            The cluster size threshold to use when outlining clusters.
        fdr_vmid : float
            The FDR value at the middle of the colorscale used for plotting the
            q-value heatmap.
        color_cycle : list of matplotlib colors
            The color cycle to use over conditions.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis, grid of pyplot axes, function
            The first pyplot axis returned is injected by ``@plotter``.
            The grid of pyplot axes is the second return value from the call to
            ``plt.subplots()`` that is used to create the grid.
            The function takes two args, an FDR and a cluster size, and redraws
            the cluster outlines using the new parameters.
        """
        # load everything
        row = np.load('%s/row_%s.npy' % (self.outdir, chrom))
        col = np.load('%s/col_%s.npy' % (self.outdir, chrom))
        raw = np.load('%s/raw_%s.npy' % (self.outdir, chrom))
        scaled = np.load('%s/scaled_%s.npy' % (self.outdir, chrom))
        disp_idx = np.load('%s/disp_idx_%s.npy' % (self.outdir, chrom))
        loop_idx = np.load('%s/loop_idx_%s.npy' % (self.outdir, chrom)) \
            if self.loop_patterns else np.ones(disp_idx.sum(), dtype=bool)
        mu_hat_alt = np.load('%s/mu_hat_alt_%s.npy' % (self.outdir, chrom))
        mu_hat_null = np.load('%s/mu_hat_null_%s.npy' % (self.outdir, chrom))
        qvalues = np.load('%s/qvalues_%s.npy' % (self.outdir, chrom))

        return plot_grid(i, j, w, row, col, raw, scaled, mu_hat_alt,
                         mu_hat_null, qvalues, disp_idx, loop_idx, self.design,
                         fdr, cluster_size, vmax=vmax, fdr_vmid=fdr_vmid,
                         color_cycle=color_cycle, despine=despine, **kwargs)
