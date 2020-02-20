import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter


@plotter
def plot_heatmap(matrix, cmap='Reds', vmin=0, vmax=100, despine=False,
                 **kwargs):
    """
    Plots a simple heatmap of a dense matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The dense matrix to visualize.
    cmap : matplotlib colormap
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
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    plt.xticks([])
    plt.yticks([])
