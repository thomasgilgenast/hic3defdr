import matplotlib.pyplot as plt

from lib5c.util.plotting import plotter

from hic3defdr.util.matrices import select_matrix


@plotter
def plot_heatmap(row, col, data, row_slice, col_slice, cmap='Reds', vmin=0,
                 vmax=100, despine=False, **kwargs):
    """
    Plots a simple heatmap of a slice of a contact matrix.

    Parameters
    ----------
    row, col, data : np.ndarray
        Sparse COO-style description of the contact matrix.
    row_slice, col_slice : slice
        The row and column slice, respectively, to plot.
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
    matrix = select_matrix(row_slice, col_slice, row, col, data)
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    plt.xticks([-0.5, matrix.shape[1]-0.5], [row_slice.start, row_slice.stop])
    plt.yticks([-0.5, matrix.shape[0]-0.5], [col_slice.start, col_slice.stop])
