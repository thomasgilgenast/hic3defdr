import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib5c.util.plotting import plotter
from lib5c.plotters.colormaps import get_colormap

from fast3defdr.matrices import select_matrix, dilate


@plotter
def plot_grid(i, j, w, row, col, raw, scaled, mu_hat_alt, mu_hat_null, qvalues,
              disp_idx, loop_idx, design, sig_cluster_csrs, insig_cluster_csrs,
              fdr, cluster_size, vmax=100, fdr_threshold=0.05, despine=False,
              **kwargs):
    # precompute some things
    max_reps = np.max(np.sum(design, axis=0))
    idx = np.where((row[disp_idx] == i) & (col[disp_idx] == j))[0][0]
    extent = [-0.5, 2 * w - 0.5, -0.5, 2 * w - 0.5]
    rs, cs = slice(i - w, i + w), slice(j - w, j + w)
    f = raw[disp_idx] / scaled[disp_idx]

    # plot
    fig, ax = plt.subplots(design.shape[1] + 1, max_reps + 1,
                           figsize=(design.shape[1] * 6, max_reps * 6))
    bwr = get_colormap('bwr', set_bad='g')
    red = get_colormap('Reds', set_bad='g')
    ax[-1, 0].imshow(
        select_matrix(
            rs, cs, row[disp_idx][loop_idx], col[disp_idx][loop_idx],
            -np.log10(qvalues)),
        cmap=bwr, interpolation='none', vmin=0, vmax=-np.log10(fdr_threshold)*2)
    ax[-1, 0].set_title('qvalues')
    for c in range(design.shape[1]):
        ax[c, 0].imshow(
            select_matrix(
                rs, cs, row[disp_idx], col[disp_idx],
                mu_hat_alt[:, np.where(design.values[:, c])[0][0]]),
            cmap=red, interpolation='none', vmin=0, vmax=vmax)
        ax[c, 0].set_ylabel(design.columns[c])
        ax_idx = 1
        for r in range(design.shape[0]):
            if not design.values[r, c]:
                continue
            ax[c, ax_idx].imshow(
                select_matrix(rs, cs, row, col, scaled[:, r]),
                cmap=red, interpolation='none', vmin=0, vmax=vmax)
            ax[c, ax_idx].set_title(design.index[r])
            ax_idx += 1
    ax[0, 0].set_title('alt model mean')
    for r in range(design.shape[1] + 1):
        for c in range(max_reps + 1):
            ax[r, c].get_xaxis().set_ticks([])
            ax[r, c].get_yaxis().set_ticks([])
            if r == design.shape[1] and c == 0:
                break

    sns.stripplot(data=[scaled[disp_idx, :][idx, design.values[:, c]]
                        for c in range(design.shape[1])], ax=ax[-1, 1])
    for c in range(design.shape[1]):
        ax[-1, 1].hlines(
            mu_hat_alt[idx, np.where(design.values[:, c])[0][0]],
            c - 0.1, c + 0.1, color='C%i' % c,
            label='alt' if c == 0 else None)
        ax[-1, 1].hlines(
            mu_hat_null[idx], c - 0.1, c + 0.1, color='C%i' % c,
            linestyles='--', label='null' if c == 0 else None)
    ax[-1, 1].set_xticklabels(design.columns.tolist())
    ax[-1, 1].set_title('normalized values')
    ax[-1, 1].set_xlabel('condition')
    ax[-1, 1].legend()
    sns.despine(ax=ax[-1, 1])

    sns.stripplot(
        data=[[raw[disp_idx, :][idx, r]]
              for r in range(design.shape[0])],
        palette=['C%i' % c for c in np.where(design)[1]], ax=ax[-1, 2])
    for r in range(design.shape[0]):
        ax[-1, 2].hlines(
            mu_hat_alt[idx, r] * f[idx, r], r - 0.1, r + 0.1,
            color='C%i' % np.where(design)[1][r],
            label='alt' if r == 0 else None)
        ax[-1, 2].hlines(
            mu_hat_null[idx] * f[idx, r], r - 0.1, r + 0.1,
            color='C%i' % np.where(design)[1][r], linestyles='--',
            label='null' if r == 0 else None)
    ax[-1, 2].set_xticklabels(design.index.tolist())
    ax[-1, 2].set_title('raw values')
    ax[-1, 2].set_xlabel('replicate')
    ax[-1, 2].legend()
    sns.despine(ax=ax[-1, 2])

    contours = []

    def outline_clusters(fdr, cluster_size):
        if contours:
            for contour in contours:
                for coll in contour.collections:
                    coll.remove()
            del contours[:]
        k = (fdr, cluster_size)
        contours.append(ax[-1, 0].contour(
            dilate(sig_cluster_csrs[k][rs, cs].toarray(), 2), [0.5],
            colors='orange', linewidths=3, extent=extent))
        contours.append(ax[-1, 0].contour(
            dilate(insig_cluster_csrs[k][rs, cs].toarray(), 2), [0.5],
            colors='gray', linewidths=3, extent=extent))
        for c in range(design.shape[1]):
            contours.append(ax[c, 0].contour(
                dilate(sig_cluster_csrs[k][rs, cs].toarray(), 2), [0.5],
                colors='purple', linewidths=3, extent=extent))
            contours.append(ax[c, 0].contour(
                dilate(insig_cluster_csrs[k][rs, cs].toarray(), 2), [0.5],
                colors='gray', linewidths=3, extent=extent))

    outline_clusters(fdr, cluster_size)

    return ax, outline_clusters
