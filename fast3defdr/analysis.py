import pickle
import glob

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import seaborn as sns

from lib5c.util.system import check_outdir
from lib5c.util.statistics import adjust_pvalues
from lib5c.util.plotting import plotter
from lib5c.plotters.colormaps import get_colormap

from fast3defdr.matrices import sparse_union, select_matrix, \
    dilate
from fast3defdr.clusters import load_clusters, save_clusters, \
    clusters_to_coo
from fast3defdr.scaling import median_of_ratios
from fast3defdr.dispersion import estimate_dispersion
from fast3defdr.lrt import lrt
from fast3defdr.thresholding import threshold_on_fdr_and_cluster
from fast3defdr.plotting.distance_dependence import plot_dd_curves
from fast3defdr.plotting.histograms import plot_pvalue_histogram
from fast3defdr.plotting.dispersion import plot_variance_fit, \
    plot_dispersion_fit


class Fast3DeFDR(object):
    """
    Main object for fast3defdr analysis.

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
        Fast3DeFDR analyses cannot co-exist in the same directory. The directory
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
    loop_patterns : list of str
        File path patterns to sparse JSON formatted cluster files representing
        called loops, in no particular order and not necessarily corresponding
        to the replicates. Each file path pattern should contain at least one
        '<chrom>' which will be replaced with the chromosome name when loading
        data for specific chromosomes.
    """

    def __init__(self, raw_npz_patterns, bias_patterns, chroms, design, outdir,
                 dist_thresh_min=4, dist_thresh_max=1000, bias_thresh=0.1,
                 mean_thresh=5, loop_patterns=None):
        """
        Base constructor. See ``help(Fast3DeFDR)`` for details.
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
        self.picklefile = '%s/pickle' % self.outdir
        check_outdir(self.picklefile)
        with open(self.picklefile, 'wb') as handle:
            pickle.dump(self, handle, -1)

    @classmethod
    def load(cls, outdir):
        """
        Loads a Fast3DeFDR analysis object from disk.

        It is safe to have multiple instances of the same analysis open at once.

        Parameters
        ----------
        outdir : str
            Folder path to where the Fast3DeFDR was saved.

        Returns
        -------
        Fast3DeFDR
            The loaded object.
        """
        with open('%s/pickle' % outdir, 'rb') as handle:
            return pickle.load(handle)

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

    def load_data(self, name, chrom):
        """
        Loads arbitrary data for one chromosome.

        Parameters
        ----------
        name : str
            The name of the data to load.
        chrom : str
            The name of the chromosome to load data for.

        Returns
        -------
        np.ndarray
            The loaded data.
        """
        return np.load('%s/%s_%s.npy' % (self.outdir, name, chrom))

    def save_data(self, data, name, chrom):
        """
        Saves arbitrary data for one chromosome to disk.

        Parameters
        ----------
        data : np.ndarray
            The data to save.
        name : str
            The name of the data to save.
        chrom : str
            The name of the chromosome to save data for.
        """
        np.save('%s/%s_%s.npy' % (self.outdir, name, chrom), data)

    def prepare_data(self, chrom):
        """
        Prepares raw and normalized data for one chromosome.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to prepare raw data for.
        """
        print('preparing data for chrom %s' % chrom)
        print('  loading bias')
        bias = self.load_bias(chrom)

        print('  computing union pixel set')
        row, col = sparse_union(
            [pattern.replace('<chrom>', chrom)
             for pattern in self.raw_npz_patterns],
            dist_thresh_min=self.dist_thresh_min,
            dist_thresh_max=self.dist_thresh_max,
            bias=bias
        )

        print('  loading raw data')
        raw = np.zeros((len(row), len(self.raw_npz_patterns)), dtype=int)
        for i, pattern in enumerate(self.raw_npz_patterns):
            raw[:, i] = sparse.load_npz(pattern.replace('<chrom>', chrom)) \
                .tocsr()[row, col]

        print('  loading balanced data')
        balanced = np.zeros((len(row), len(self.raw_npz_patterns)), dtype=float)
        for r, pattern in enumerate(self.raw_npz_patterns):
            balanced[:, r] = sparse.load_npz(pattern.replace('<chrom>', chrom)) \
                                 .tocsr()[row, col] / (
                                         bias[row, r] * bias[col, r])

        print('  computing size factors')
        size_factors = median_of_ratios(balanced)
        scaled = balanced / size_factors

        print('  saving data to disk')
        self.save_data(row, 'row', chrom)
        self.save_data(col, 'col', chrom)
        self.save_data(raw, 'raw', chrom)
        self.save_data(size_factors, 'size_factors', chrom)
        self.save_data(scaled, 'scaled', chrom)

    def estimate_disp(self, chrom, estimator='cml', trend='mean', n_bins=100):
        """
        Estimates dispersion for one chromosome.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to prepare data for.
        estimator : 'cml', 'mme', or a function
            Pass 'cml' or 'mme' to use conditional maximum likelihood or method
            of moments estimation, respectively, to estimate the dispersion
            within each bin. Pass a function that takes in a
            (pixels, replicates) shaped array of data and returns a dispersion
            value to use that instead.
        trend : 'mean' or 'dist'
            Whether to estimate the dispersion trend with respect to mean or
            interaction distance.
        n_bins : int
            Number of bins to use during dispersion estimation.
        """
        print('estimating dispersion for chrom %s' % chrom)
        print('  loading scaled data')
        scaled = self.load_data('scaled', chrom)

        print('  computing pixel-wise mean per condition')
        mean = np.dot(scaled, self.design) / np.sum(self.design, axis=0).values
        disp_idx = np.any(mean > self.mean_thresh, axis=1)

        if trend == 'dist':
            row = self.load_data('row', chrom)[disp_idx]
            col = self.load_data('col', chrom)[disp_idx]
            dist = col - row
            cov = np.broadcast_to(dist, (disp_idx.sum(), self.design.shape[1]))
        elif trend == 'mean':
            cov = mean[disp_idx]
        else:
            raise ValueError('trend must be \'mean\' or \'dist\'')

        disp = np.zeros((disp_idx.sum(), self.design.shape[1]))
        cov_per_bin = np.zeros((n_bins, self.design.shape[1]))
        disp_per_bin = np.zeros((n_bins, self.design.shape[1]))
        for c, cond in enumerate(self.design.columns):
            print('  estimating dispersion for condition %s' % cond)
            disp[:, c], cov_per_bin[:, c], disp_per_bin[:, c] = \
                estimate_dispersion(
                    scaled[disp_idx, :][:, self.design[cond]],
                    cov=cov[:, c],
                    estimator=estimator,
                    n_bins=n_bins,
                    logx=True if trend == 'mean' else False
            )

        print('  saving estimated dispersions to disk')
        self.save_data(disp_idx, 'disp_idx', chrom)
        self.save_data(disp, 'disp', chrom)
        self.save_data(cov_per_bin, 'cov_per_bin', chrom)
        self.save_data(disp_per_bin, 'disp_per_bin', chrom)

    def lrt(self, chrom):
        print('running LRT for chrom %s' % chrom)
        print('  loading data')
        bias = self.load_bias(chrom)
        size_factors = self.load_data('size_factors', chrom)
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        raw = self.load_data('raw', chrom)
        disp_idx = self.load_data('disp_idx', chrom)
        disp = self.load_data('disp', chrom)

        print('  computing LRT results')
        f = bias[row][disp_idx] * bias[col][disp_idx] * size_factors
        pvalues, llr, mu_hat_null, mu_hat_alt = lrt(
            raw[disp_idx, :], f, disp, self.design.values)

        if self.loop_patterns:
            print('  making loop_idx')
            loop_pixels = set.union(
                *sum((load_clusters(pattern.replace('<chrom>', chrom))
                      for pattern in self.loop_patterns), []))
            loop_idx = np.array([True if pixel in loop_pixels else False
                                 for pixel in zip(row[disp_idx],
                                                  col[disp_idx])])
            self.save_data(loop_idx, 'loop_idx', chrom)

        print('  saving results to disk')
        self.save_data(pvalues, 'pvalues', chrom)
        self.save_data(llr, 'llr', chrom)
        self.save_data(mu_hat_null, 'mu_hat_null', chrom)
        self.save_data(mu_hat_alt, 'mu_hat_alt', chrom)

    def process_chrom(self, chrom):
        """
        Processes one chromosome through p-values.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to process.
        """
        self.prepare_data(chrom)
        self.estimate_disp(chrom)
        self.lrt(chrom)

    def process_all(self):
        """
        Processes all chromosomes through p-values in series.
        """
        for chrom in self.chroms:
            self.process_chrom(chrom)

    def bh(self):
        """
        Applies BH-FDR control to p-values across all chromosomes to obtain
        q-values.

        Should only be run after all chromosomes have been processed through
        p-values.
        """
        print('applying BH-FDR correction')
        if self.loop_patterns:
            pvalues = [self.load_data('pvalues', chrom)[
                           self.load_data('loop_idx', chrom)]
                       for chrom in self.chroms]
        else:
            pvalues = [self.load_data('pvalues', chrom)
                       for chrom in self.chroms]
        all_qvalues = adjust_pvalues(np.concatenate(pvalues))
        offset = 0
        for i, chrom in enumerate(self.chroms):
            self.save_data(all_qvalues[offset:offset + len(pvalues[i])],
                           'qvalues', chrom)
            offset += len(pvalues[i])

    def threshold_chrom(self, chrom, fdr=0.05, cluster_size=4):
        """
        Thresholds and clusters significantly differential pixels on one
        chromosome.

        Should only be run after q-values have been obtained.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to threshold.
        fdr : float or list of float
            The FDR to threshold on. Pass a list to do a sweep in series.
        cluster_size : int or list of int
            Clusters smaller than this size will be filtered out. Pass a list to
            do a sweep in series.
        """
        print('thresholding and clustering chrom %s' % chrom)
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
            sig_clusters, insig_clusters = threshold_on_fdr_and_cluster(
                qvalues, row, col, fdr)
            for s in cluster_size:
                # threshold on cluster size and save to disk
                save_clusters([c for c in sig_clusters if len(c) > s],
                              '%s/sig_%s_%g_%i.json' %
                              (self.outdir, chrom, f, s))
                save_clusters([c for c in insig_clusters if len(c) > s],
                              '%s/insig_%s_%g_%i.json' %
                              (self.outdir, chrom, f, s))

    def threshold_all(self, fdr=0.05, cluster_size=4):
        """
        Thresholds and clusters significantly differential pixels on all
        chromosomes in series.

        Should only be run after q-values have been obtained.

        Parameters
        ----------
        fdr : float or list of float
            The FDR to threshold on. Pass a list to do a sweep in series.
        cluster_size : int or list of int
            Clusters smaller than this size will be filtered out. Pass a list to
            do a sweep in series.
        """
        for chrom in self.chroms:
            self.threshold_chrom(chrom, fdr=fdr, cluster_size=cluster_size)

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

        # compute dist
        dist = col - row

        return plot_dd_curves(balanced, scaled, dist, self.design, log=log,
                              **kwargs)

    def plot_dispersion_fit(self, chrom, cond, yaxis='disp', **kwargs):
        """
        Plots a hexbin plot of pixel-wise mean vs either dispersion or variance,
        overlaying the Poisson line and the mean-variance relationship
        represented by the fitted dispersions.

        Parameters
        ----------
        chrom, cond : str
            The name of the chromosome and condition, respectively, to plot the
            fit for.
        yaxis : 'disp' or 'var'
            What to plot on the y-axis.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # establish plot_fn
        if yaxis == 'disp':
            plot_fn = plot_dispersion_fit
        elif yaxis == 'var':
            plot_fn = plot_variance_fit
        else:
            raise ValueError('yaxis must be \'disp\' or \'var\'')

        # identify cond_idx
        cond_idx = self.design.columns.tolist().index(cond)

        # load everything
        disp_idx = self.load_data('disp_idx', chrom)
        scaled = self.load_data('scaled', chrom)[disp_idx, self.design[cond]]
        disp = self.load_data('disp', chrom)[:, cond_idx]
        cov_per_bin = self.load_data('cov_per_bin', chrom)[:, cond_idx]
        disp_per_bin = self.load_data('disp_per_bin', chrom)[:, cond_idx]

        # compute mean and sample variance
        mean = np.mean(scaled, axis=1)
        var = np.var(scaled, ddof=1, axis=1)

        return plot_fn(mean, var, disp, cov_per_bin, disp_per_bin, **kwargs)

    @plotter
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
            pvalues = [np.load('%s/pvalues_%s.npy' % (self.outdir, chrom))[
                           np.load('%s/loop_idx_%s.npy' % (self.outdir, chrom))]
                       for chrom in self.chroms]
        elif idx == 'disp':
            pvalues = [np.load('%s/pvalues_%s.npy' % (self.outdir, chrom))
                       for chrom in self.chroms]
        else:
            raise ValueError('idx must be loop or disp')

        # plot
        return plot_pvalue_histogram(np.concatenate(pvalues), **kwargs)

    @plotter
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
        qvalues = [np.load('%s/qvalues_%s.npy' % (self.outdir, chrom))
                   for chrom in self.chroms]

        # plot
        return plot_pvalue_histogram(
            np.concatenate(qvalues), xlabel='qvalue', **kwargs)

    @plotter
    def plot_grid(self, chrom, i, j, w, vmax=100, fdr=0.05, cluster_size=4,
                  **kwargs):
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

        # TODO: search for and preload clusters
        n = max(row.max(), col.max())
        filenames = glob.glob('%s/sig_%s_*.json' % (self.outdir, chrom))

        # precompute some things
        f = bias[row][disp_idx] * bias[col][disp_idx] * size_factors
        max_reps = np.max(np.sum(self.design, axis=0))
        idx = np.where((row[disp_idx] == i) & (col[disp_idx] == j))[0][0]
        extent = [-0.5, 2 * w - 0.5, -0.5, 2 * w - 0.5]
        rs, cs = slice(i - w, i + w), slice(j - w, j + w)

        # plot
        fig, ax = plt.subplots(self.design.shape[1] + 1, max_reps + 1,
                               figsize=(self.design.shape[1] * 6, max_reps * 6))
        bwr = get_colormap('bwr', set_bad='g')
        red = get_colormap('Reds', set_bad='g')
        ax[-1, 0].imshow(
            select_matrix(
                rs, cs, row[disp_idx][loop_idx], col[disp_idx][loop_idx],
                -np.log10(qvalues)),
            cmap=bwr, interpolation='none', vmin=0, vmax=-np.log10(fdr) * 2)
        ax[-1, 0].set_title('qvalues')
        for c in range(self.design.shape[1]):
            ax[c, 0].imshow(
                select_matrix(
                    rs, cs, row[disp_idx], col[disp_idx],
                    mu_hat_alt[:, np.where(self.design.values[:, c])[0][0]]),
                cmap=red, interpolation='none', vmin=0, vmax=vmax)
            ax[c, 0].set_ylabel(self.design.columns[c])
            ax_idx = 1
            for r in range(self.design.shape[0]):
                if not self.design.values[r, c]:
                    continue
                ax[c, ax_idx].imshow(
                    select_matrix(rs, cs, row, col, scaled[:, r]),
                    cmap=red, interpolation='none', vmin=0, vmax=vmax)
                ax[c, ax_idx].set_title(self.design.index[r])
                ax_idx += 1
        ax[0, 0].set_title('alt model mean')
        for r in range(self.design.shape[1] + 1):
            for c in range(max_reps + 1):
                ax[r, c].get_xaxis().set_ticks([])
                ax[r, c].get_yaxis().set_ticks([])
                if r == self.design.shape[1] and c == 0:
                    break

        sns.stripplot(data=[scaled[disp_idx, :][idx, self.design.values[:, c]]
                            for c in range(self.design.shape[1])], ax=ax[-1, 1])
        for c in range(self.design.shape[1]):
            ax[-1, 1].hlines(
                mu_hat_alt[idx, np.where(self.design.values[:, c])[0][0]],
                c - 0.1, c + 0.1, color='C%i' % c,
                label='alt' if c == 0 else None)
            ax[-1, 1].hlines(
                mu_hat_null[idx], c - 0.1, c + 0.1, color='C%i' % c,
                linestyles='--', label='null' if c == 0 else None)
        ax[-1, 1].set_xticklabels(self.design.columns.tolist())
        ax[-1, 1].set_title('normalized values')
        ax[-1, 1].set_xlabel('condition')
        ax[-1, 1].legend()
        sns.despine(ax=ax[-1, 1])

        sns.stripplot(
            data=[[raw[disp_idx, :][idx, r]]
                  for r in range(self.design.shape[0])],
            palette=['C%i' % c for c in np.where(self.design)[1]], ax=ax[-1, 2])
        for r in range(self.design.shape[0]):
            ax[-1, 2].hlines(
                mu_hat_alt[idx, r] * f[idx, r], r - 0.1, r + 0.1,
                color='C%i' % np.where(self.design)[1][r],
                label='alt' if r == 0 else None)
            ax[-1, 2].hlines(
                mu_hat_null[idx] * f[idx, r], r - 0.1, r + 0.1,
                color='C%i' % np.where(self.design)[1][r], linestyles='--',
                label='null' if r == 0 else None)
        ax[-1, 2].set_xticklabels(self.design.index.tolist())
        ax[-1, 2].set_title('raw values')
        ax[-1, 2].set_xlabel('replicate')
        ax[-1, 2].legend()
        sns.despine(ax=ax[-1, 2])

        contours = []

        def outline_clusters(fdr, cluster_size):
            sig_cluster_csr = clusters_to_coo(
                load_clusters('%s/sig_%s_%g_%i.json' %
                              (self.outdir, chrom, fdr, cluster_size)),
                (bias.shape[0], bias.shape[0])).tocsr()
            insig_cluster_csr = clusters_to_coo(
                load_clusters('%s/insig_%s_%g_%i.json' %
                              (self.outdir, chrom, fdr, cluster_size)),
                (bias.shape[0], bias.shape[0])).tocsr()
            if contours:
                for contour in contours:
                    for coll in contour.collections:
                        coll.remove()
                del contours[:]
            contours.append(ax[-1, 0].contour(
                dilate(sig_cluster_csr[rs, cs].toarray(), 2), [0.5],
                colors='orange', linewidths=3, extent=extent))
            contours.append(ax[-1, 0].contour(
                dilate(insig_cluster_csr[rs, cs].toarray(), 2), [0.5],
                colors='gray', linewidths=3, extent=extent))
            for c in range(self.design.shape[1]):
                contours.append(ax[c, 0].contour(
                    dilate(sig_cluster_csr[rs, cs].toarray(), 2), [0.5],
                    colors='purple', linewidths=3, extent=extent))
                contours.append(ax[c, 0].contour(
                    dilate(insig_cluster_csr[rs, cs].toarray(), 2), [0.5],
                    colors='gray', linewidths=3, extent=extent))

        outline_clusters(fdr, cluster_size)

        return ax, outline_clusters
