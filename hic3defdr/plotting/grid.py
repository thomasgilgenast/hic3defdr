import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib5c.util.plotting import plotter
from lib5c.plotters.colormaps import get_colormap

from hic3defdr.util.matrices import dilate, select_matrix
from hic3defdr.util.thresholding import threshold_and_cluster, size_filter
from hic3defdr.util.clusters import clusters_to_coo
from hic3defdr.util.classification import classify
from hic3defdr.plotting.heatmap import plot_heatmap


@plotter
def plot_grid(i, j, w, row, col, raw, scaled, mu_hat_alt, mu_hat_null, qvalues,
              disp_idx, loop_idx, design, fdr, cluster_size, vmax=100,
              fdr_vmid=0.05,
              color_cycle=('blue', 'green', 'purple', 'yellow', 'cyan', 'red'),
              despine=False, **kwargs):
    """
    Plots a combination visualization grid focusing on a specific pixel on a
    specific chromosome, combining heatmaps, cluster outlines, and stripplots.

    Parameters
    ----------
    i, j : int
        The row and column index of the pixel to focus on.
    w : int
        The size of the heatmap will be ``2*w + 1`` bins in each dimension.
    row, col : np.ndarray
        The row and column indices corresponding to the rows of the ``raw`` and
        ``scaled`` matrices.
    raw, scaled : np.ndarray
        The raw and scaled data for each pixel (rows) and each replicate
        (columns).
    mu_hat_alt, mu_hat_null : np.ndarray
        The estimated mean parameter under the alternate and null model,
        respectively. First dimension is pixels for which dispersion was
        estimated, whose row and column coordinates in the complete square
        matrix are given by ``row[disp_idx]`` and ``col[disp_idx]``,
        respectively. Columns of ``mu_hat_alt`` correspond to conditions, while
        ``mu_hat_null`` has no second dimension.
    qvalues : np.ndarray
        Vector of q-values called per pixel whose dispersion was estimated and
        which lies in a loop. The row and column coordinates in the complete
        square matrix are given by ``row[disp_idx][loop_idx]`` and
        ``col[disp_idx][loop_idx]``, respectively.
    disp_idx : np.ndarray
        Boolean matrix indicating which pixels in ``zip(row, col)`` had their
        dispersion estimated.
    loop_idx : np.ndarray
        Boolean matrix indicating which pixels in
        ``zip(row[disp_idx], col[disp_idx])`` lie within loops.
    design : pd.DataFrame
        Pass a DataFrame with boolean dtype whose rows correspond to replicates
        and whose columns correspond to conditions. Replicate and condition
        names will be inferred from the row and column labels, respectively.
    fdr : float
        The FDR threshold to use when outlining clusters.
    cluster_size : int
        The cluster size threshold to use when outlining clusters.
    vmax : float
        The maximum of the colorscale to use when plotting normalized
        heatmaps.
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
        The first pyplot axis returned is injected by ``@plotter``. The grid of
        pyplot axes is the second return value from the call to
        ``plt.subplots()`` that is used to create the grid. The function takes
        two args, an FDR and a cluster size, and redraws the cluster outlines
        using the new parameters.
    """
    # precompute some things
    max_reps = np.max(np.sum(design, axis=0))
    idx = np.where((row[disp_idx] == i) & (col[disp_idx] == j))[0][0]
    extent = [-0.5, 2 * w + 0.5, -0.5, 2 * w + 0.5]
    rs, cs = slice(i - w, i + w + 1), slice(j - w, j + w + 1)
    f = raw[disp_idx] / scaled[disp_idx]
    n = max(row.max(), col.max()) + 1
    mu_hat_alt = np.dot(mu_hat_alt, design.values.T)
    mean = np.dot(scaled, design) / np.sum(design, axis=0).values

    # set the color cycle over conditions
    cc = color_cycle

    # plot
    fig, ax = plt.subplots(design.shape[1] + 1, max_reps + 1,
                           figsize=(design.shape[1] * 6, max_reps * 6))
    bwr = get_colormap('bwr', set_bad='lightgray')
    red = get_colormap('Reds', set_bad='lightgray')
    plot_heatmap(
        select_matrix(rs, cs, row[disp_idx][loop_idx], col[disp_idx][loop_idx],
                      -np.log10(qvalues)),
        cmap=bwr, vmax=-np.log10(fdr_vmid)*2, ax=ax[-1, 0]
    )
    ax[-1, 0].set_title('qvalues')
    for c in range(design.shape[1]):
        plot_heatmap(
            select_matrix(rs, cs, row, col, mean[:, c]),
            cmap=red, vmax=vmax, ax=ax[c, 0]
        )
        ax[c, 0].set_ylabel(design.columns[c])
        ax_idx = 1
        for r in range(design.shape[0]):
            if not design.values[r, c]:
                continue
            plot_heatmap(
                select_matrix(rs, cs, row, col, scaled[:, r]),
                cmap=red, vmax=vmax, ax=ax[c, ax_idx]
            )
            ax[c, ax_idx].set_title(design.index[r])
            ax_idx += 1
    ax[0, 0].set_title('mean')
    for r in range(design.shape[1] + 1):
        for c in range(max_reps + 1):
            if r < design.shape[1] - 1:
                ax[r, c].get_xaxis().set_ticks([])
            if c > 0:
                ax[r, c].get_yaxis().set_ticks([])
            if r == design.shape[1] and c == 0:
                break

    sns.stripplot(
        data=[scaled[disp_idx, :][idx, design.values[:, c]]
              for c in range(design.shape[1])],
        palette=[cc[c % len(cc)] for c in range(design.shape[1])],
        ax=ax[-1, 1]
    )
    for c in range(design.shape[1]):
        ax[-1, 1].hlines(
            mu_hat_alt[idx, np.where(design.values[:, c])[0][0]],
            c - 0.1, c + 0.1, color=cc[c % len(cc)],
            label='alt' if c == 0 else None)
        ax[-1, 1].hlines(
            mu_hat_null[idx], c - 0.1, c + 0.1, color=cc[c % len(cc)],
            linestyles='--', label='null' if c == 0 else None)
    ax[-1, 1].set_xticklabels(design.columns.tolist())
    ax[-1, 1].set_title('normalized values')
    ax[-1, 1].set_xlabel('condition')
    ax[-1, 1].legend()
    sns.despine(ax=ax[-1, 1])

    sns.stripplot(
        data=[[raw[disp_idx, :][idx, r]]
              for r in range(design.shape[0])],
        palette=[cc[c % len(cc)] for c in np.where(design)[1]],
        ax=ax[-1, 2]
    )
    for r in range(design.shape[0]):
        ax[-1, 2].hlines(
            mu_hat_alt[idx, r] * f[idx, r], r - 0.1, r + 0.1,
            color=cc[np.where(design)[1][r] % len(cc)],
            label='alt' if r == 0 else None)
        ax[-1, 2].hlines(
            mu_hat_null[idx] * f[idx, r], r - 0.1, r + 0.1,
            color=cc[np.where(design)[1][r] % len(cc)], linestyles='--',
            label='null' if r == 0 else None)
    ax[-1, 2].set_xticklabels(design.index.tolist())
    ax[-1, 2].set_title('raw values')
    ax[-1, 2].set_xlabel('replicate')
    ax[-1, 2].legend()
    sns.despine(ax=ax[-1, 2])

    contours = []
    clusters = {}

    def outline_clusters(fdr, cluster_size):
        if fdr not in clusters:
            clusters[fdr] = {}
            clusters[fdr]['base'] = threshold_and_cluster(
                qvalues, row[disp_idx][loop_idx], col[disp_idx][loop_idx], fdr)
        if cluster_size not in clusters[fdr]:
            clusters[fdr][cluster_size] = dict(zip(
                range(design.shape[1]),
                map(lambda x: clusters_to_coo(size_filter(x, cluster_size),
                                              (n, n)).tocsr()[rs, cs].toarray(),
                    classify(
                        row[disp_idx][loop_idx],
                        col[disp_idx][loop_idx],
                        mu_hat_alt[loop_idx],
                        size_filter(clusters[fdr]['base'][0], cluster_size)
                ))
            ))
            clusters[fdr][cluster_size]['insig'] = clusters_to_coo(
                size_filter(clusters[fdr]['base'][1], cluster_size),
                (n, n)).tocsr()[rs, cs].toarray()
        if contours:
            for contour in contours:
                for coll in contour.collections:
                    coll.remove()
            del contours[:]
        for c in range(design.shape[1]):
            contour = ax[-1, 0].contour(
                dilate(clusters[fdr][cluster_size][c], 2),
                [0.5], colors=cc[c % len(cc)], linewidths=3, extent=extent)
            contour.collections[0].set_label(design.columns[c])
            contours.append(contour)
        contour = ax[-1, 0].contour(
            dilate(clusters[fdr][cluster_size]['insig'], 2),
            [0.5], colors='gray', linewidths=3, extent=extent)
        contour.collections[0].set_label('constitutive')
        contours.append(contour)
        for cond_idx in range(design.shape[1]):
            for c in range(design.shape[1]):
                contour = ax[cond_idx, 0].contour(
                    dilate(clusters[fdr][cluster_size][c], 2),
                    [0.5], colors=cc[c % len(cc)], linewidths=3, extent=extent)
                contour.collections[0].set_label(design.columns[c])
                contours.append(contour)
            contour = ax[cond_idx, 0].contour(
                dilate(clusters[fdr][cluster_size]['insig'], 2),
                [0.5], colors='gray', linewidths=3, extent=extent)
            contour.collections[0].set_label('constitutive')
            contours.append(contour)

    outline_clusters(fdr, cluster_size)
    ax[-1, 0].legend(loc='lower left')

    return ax, outline_clusters
