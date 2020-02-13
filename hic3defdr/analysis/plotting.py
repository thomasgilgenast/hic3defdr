import numpy as np

from lib5c.algorithms.correlation import \
    make_pairwise_correlation_matrix_from_counts_matrix
from lib5c.plotters.correlation import plot_correlation_matrix

from hic3defdr.plotting.distance_dependence import plot_dd_curves
from hic3defdr.plotting.histograms import plot_pvalue_histogram
from hic3defdr.plotting.dispersion import plot_mvr, plot_ddr
from hic3defdr.plotting.ma import plot_ma
from hic3defdr.plotting.grid import plot_grid
from hic3defdr.plotting.heatmap import plot_heatmap


class PlottingHiC3DeFDR(object):
    """
    Mixin class containing plotting functions for HiC3DeFDR.
    """
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

        return plot_dd_curves(row, col, balanced, scaled,
                              repnames=self.design.index, log=log, **kwargs)

    def plot_dispersion_fit(self, cond, xaxis='dist', yaxis='disp',
                            dist_max=None, scatter_fit=-1, scatter_size=36,
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
            in bin units. Pass None to use ``self.dist_thresh_max``.
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
        # short circuit to plot_ddr() if possible
        if xaxis == 'dist' and yaxis == 'disp' and scatter_fit == -1 \
                and distance is None and hexbin is False and logx is False \
                and logy is False:
            return self.plot_ddr(cond, dist_max=dist_max,
                                 scatter_size=scatter_size, **kwargs)

        # resolve max_dist
        if dist_max is None:
            dist_max = self.dist_thresh_max

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

    def plot_ddr(self, cond, dist_max=None, scatter_size=36, **kwargs):
        """
        Fast alternative to plot_dispersion_fit() that only supports plotting
        distance versus dispersion, with no hexbin or ``scatter_points``
        support.

        Parameters
        ----------
        cond : str
            The name of the chromosome and condition, respectively, to plot the
            fit for.
        dist_max : int
            If ``xaxis`` is 'dist', the maximum distance to include on the plot
            in bin units. Pass None to use ``self.dist_thresh_max``.
        scatter_size : int
            The marker size when plotting scatterplots.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # resolve max_dist
        if dist_max is None:
            dist_max = self.dist_thresh_max

        # identify cond_idx
        cond_idx = self.design.columns.tolist().index(cond)

        # load everything
        disp_per_dist = self.load_data('disp_per_dist')[:, cond_idx]
        idx = np.isfinite(disp_per_dist)
        disp_per_bin = disp_per_dist[idx]
        dist_per_bin = np.arange(self.dist_thresh_max + 1)[idx]
        disp_fn = self.load_disp_fn(cond)

        # plot
        return plot_ddr(dist_per_bin, disp_per_bin, disp_fn,
                        scatter_size=scatter_size, **kwargs)

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

    def plot_ma(self, fdr=0.05, conds=None, include_non_loops=True, s=-1,
                nonloop_s=None, density_dpi=72, vmax=None, nonloop_vmax=None,
                ax=None, legend=True, **kwargs):
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
            The marker size to use for the scatterplot, or -1 to use a
            scatter density plot.
        nonloop_s : float, optional
            Pass a separate marker size to use specifically for the non-loop
            pixels if `include_non_loops` is True. Useful for drawing just the
            non-loop pixels as a density by passing `s=1, nonloop_s=-1`. Pass
            None to use `s` as the size for both loop and non-loop pixels.
        density_dpi : int
            If `s` or `nonloop_s` are -1 this specifies the DPI to use for the
            density grid.
        vmax, nonloop_vmax : float, optional
            The vmax to use for `ax.scatter_density()` if `s` or `nonloop_s` is
            -1, respectively. Pass None to choose values automatically.
        ax : pyplot axis
            The axis to plot to. Must have been created with
            `projection='scatter_density'`. Pass None to create a new axis.
        legend : bool
            Pass True to add a legend. Note that passing `legend='outside'` is
            not supported.
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

        # stuff common kwargs into kwargs
        kwargs['names'] = conds
        kwargs['s'] = s
        kwargs['nonloop_s'] = nonloop_s
        kwargs['density_dpi'] = density_dpi
        kwargs['vmax'] = vmax
        kwargs['nonloop_vmax'] = vmax
        kwargs['ax'] = ax
        kwargs['legend'] = legend

        # plot
        if include_non_loops:
            plot_ma(mean, sig_idx, loop_idx=loop_idx, **kwargs)
        else:
            plot_ma(mean[loop_idx], sig_idx, **kwargs)

    def plot_correlation_matrix(self, stage='scaled', idx='loop',
                                correlation='spearman', colorscale=(0.75, 1.0),
                                **kwargs):
        """
        Plots a matrix of pairwise correlations among all replicates.

        Parameters
        ----------
        stage : {'raw', 'scaled'}
            Specify the stage of the data to compute correlations between.
        idx : {'disp', 'loop'}
            Pass 'disp' to compute correlations for all points for which
            dispersion was estimated. Pass 'loop' to compute correlations for
            all points which are in loops (available only if ``loop_patterns``
            was passed to the constructor).
        correlation : {'spearman', 'pearson'}
            Which correlation coefficient to compute.
        colorscale : tuple of float
            The min and max values of the correlation to use to color the
            matrix.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        # resolve idx
        if idx == 'disp':
            idx, _ = self.load_data('disp_idx', 'all')
        elif idx == 'loop':
            idx = (
                self.load_data('disp_idx', 'all')[0],
                self.load_data('loop_idx', 'all')[0]
            )
        else:
            raise ValueError('idx must be \'disp\' or \'loop\'')

        # load data
        data = self.load_data(stage, 'all', idx=idx)[0].T

        # plot
        return plot_correlation_matrix(
            make_pairwise_correlation_matrix_from_counts_matrix(
                data, correlation=correlation
            ),
            label_values=self.design.index.tolist(),
            colorscale=colorscale,
            **kwargs
        )

    def plot_heatmap(self, rep, chrom, row_slice, col_slice, stage='scaled',
                     cmap='Reds', vmin=0, vmax=100, **kwargs):
        """
        Plots a simple heatmap of a slice of the contact matrix.

        Parameters
        ----------
        rep : str, optional
            The rep to plot. Ignored if ``stage`` is 'qvalues'.
        chrom : str
            The chromosome to plot.
        row_slice, col_slice : slice
            The row and column slice, respectively, to plot.
        stage : {'raw', 'scaled', 'qvalues'}
            The stage of the data to plot.
        cmap : matplotlib colormap or dict
            The colormap to use for the heatmap.
        vmin, vmax : float
            The vmin and vmax to use for the heatmap colorscale.
        kwargs : kwargs
            Typical plotter kwargs.

        Returns
        -------
        pyplot axis
            The axis plotted on.
        """
        if stage in ['raw', 'scaled']:
            rep_idx = self.design.index.tolist().index(rep)
            row = self.load_data('row', chrom)
            col = self.load_data('col', chrom)
            data = self.load_data(stage, chrom)[:, rep_idx]
        elif stage == 'qvalues':
            disp_idx = self.load_data('disp_idx', chrom)
            loop_idx = self.load_data('loop_idx', chrom)
            row = self.load_data('row', chrom, idx=(disp_idx, loop_idx))
            col = self.load_data('col', chrom, idx=(disp_idx, loop_idx))
            data = self.load_data('qvalues', chrom)
        plot_heatmap(row, col, data, row_slice, col_slice, cmap=cmap, vmin=vmin,
                     vmax=vmax, **kwargs)

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
            The size of the heatmap will be ``2*w + 1`` bins in each dimension.
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
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        raw = self.load_data('raw', chrom)
        scaled = self.load_data('scaled', chrom)
        disp_idx = self.load_data('disp_idx', chrom)
        loop_idx = self.load_data('loop_idx', chrom) \
            if self.loop_patterns else np.ones(disp_idx.sum(), dtype=bool)
        mu_hat_alt = self.load_data('mu_hat_alt', chrom)
        mu_hat_null = self.load_data('mu_hat_null', chrom)
        qvalues = self.load_data('qvalues', chrom)

        return plot_grid(i, j, w, row, col, raw, scaled, mu_hat_alt,
                         mu_hat_null, qvalues, disp_idx, loop_idx, self.design,
                         fdr, cluster_size, vmax=vmax, fdr_vmid=fdr_vmid,
                         color_cycle=color_cycle, despine=despine, **kwargs)
