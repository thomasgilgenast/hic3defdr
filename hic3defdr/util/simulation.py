import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.ndimage as ndimage

from lib5c.util.distributions import freeze_distribution

from hic3defdr.util.scaled_nb import mvr
from hic3defdr.util.printing import eprint


def perturb_cluster(matrix, cluster, effect, respect_zeros=True):
    """
    Perturbs a specific cluster in a contact matrix with a given effect.

    Operates in-place.

    Based on a notebook linked here:
    https://colab.research.google.com/drive/1dk9kX57ZtlxQ3jubrKL_q2r8LZnSlVwY

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        The contact matrix. Must support slicing.
    cluster : list of tuple of int
        A list of (i, j) tuples marking the position of points which belong to
        the cluster which we want to perturb.
    effect : float
        The effect to apply to the cluster. Values in ``matrix`` under the
        cluster footprint will be shifted by this proportion of their original
        value.
    respect_zeros : bool
        Pass True to preserve the sparsity structure of ``matrix`` if it is
        sparse. Has no effect if ``matrix`` is dense.
    """
    # come up with a rectangle that covers the cluster with a 1px buffer
    rs, cs = map(np.array, zip(*cluster))
    r_min = max(np.min(rs) - 1, 0)
    r_max = min(np.max(rs) + 1, matrix.shape[0] - 1)
    c_min = max(np.min(cs) - 1, 0)
    c_max = min(np.max(cs) + 1, matrix.shape[1] - 1)
    r_size = r_max - r_min + 1
    c_size = c_max - c_min + 1
    r_slice = slice(r_min, r_max + 1)
    c_slice = slice(c_min, c_max + 1)

    # build up the effect footprint
    footprint = np.zeros((r_size, c_size), dtype=float)
    footprint[rs - r_min, cs - c_min] = 1
    struct = ndimage.generate_binary_structure(2, 2)
    footprint += ndimage.binary_dilation(
        footprint, structure=struct)
    footprint /= 2

    # apply the effect footprint
    if isinstance(matrix, sparse.spmatrix) and respect_zeros:
        s = matrix[r_slice, c_slice]
        s_coo = s.tocoo()
        r_read_idx = s_coo.row
        c_read_idx = s_coo.col
        r_write_idx = r_read_idx + r_min
        c_write_idx = c_read_idx + c_min
        new_values = s.toarray() * footprint * effect
        matrix[r_write_idx, c_write_idx] += new_values[r_read_idx, c_read_idx]
    else:
        matrix[r_slice, c_slice] += matrix[r_slice, c_slice].toarray() * \
            footprint * effect


def simulate(row, col, mean, disp_fn, bias, size_factors, clusters, beta=0.5,
             p_diff=0.4, trend='mean', verbose=True):
    """
    Simulates raw contact matrices based on ``mean`` and ``disp_fn`` using
    ``bias`` and ``size_factors`` per simulated replicate and perturbing the
    loops specified in ``clusters`` with an effect size of ``beta`` and
    direction chosen at random for ``p_diff`` fraction of clusters.

    Parameters
    ----------
    row, col : np.ndarray
        Row and column indices identifying the location of pixels in ``mean``.
    mean : np.ndarray
        Vector of mean values for each pixel to use as a base to simulate from.
    disp_fn : function
        Function that returns a dispersion given a mean or distance (as
        specified by ``trend``). Will be used to determine the dispersion
        values to use during simulation.
    bias : np.ndarray
        Rows are bins of the full contact matrix, columns are to-be-simulated
        replicates. Each column represents the bias vector to use for simulating
        that replicate.
    size_factors : np.ndarray
        Vector of size factors to use for simulating for each to-be-simulated
        replicate. To use a different size factor at different distance scales,
        pass a matrix whose rows correspond to distance scales and whose columns
        correspond to replicates.
    clusters : list of list of tuple
        The outer list is a list of clusters which represent the locations of
        loops. Each cluster is a list of (i, j) tuples marking the position of
        pixels which belong to that cluster.
    beta : float
        The effect size of the loop perturbations to use when simulating.
        Perturbed loops will be strengthened or weakened by this fraction of
        their original strength.
    p_diff : float or list of float
        Pass a single float to specify the probability that a loop will be
        perturbed across the simulated conditions. Pass four floats to specify
        the probabilities of all four specific perturbations: up in A, down in
        A, up in B, down in B. The remaining loops will be constitutive.
    trend : 'mean' or 'dist'
        Whether ``disp_fn`` returns the smoothed dispersion as a function of
        mean or of interaction distance.
    verbose : bool
        Pass False to silence reporting of progress to stderr.

    Returns
    -------
    classes : np.ndarray
        Vector of ground-truth class labels used for simulation with 'U7'
        dtype.
    gen : generator of ``scipy.sparse.csr_matrix``
        Generates the simulated raw contact matrices for each simulated
        replicate, in order.
    """
    eprint('  assigning cluster classes', skip=not verbose)
    p = [1 - p_diff, p_diff/4, p_diff/4, p_diff/4, p_diff/4] \
        if type(p_diff) == float else [1 - sum(p_diff)] + list(p_diff)
    classes = np.random.choice(
        np.array(['constit', 'up A', 'down A', 'up B', 'down B'], dtype='U7'),
        size=len(clusters), p=p)

    # adjust indexing to make mean nonzero
    nonzero_idx = mean > 0
    row = row[nonzero_idx]
    col = col[nonzero_idx]
    mean = mean[nonzero_idx]
    assert np.all(mean > 0)

    eprint('  perturbing clusters', skip=not verbose)
    mean_a_csr = sparse.coo_matrix(
        (mean, (row, col)), shape=(bias.shape[0], bias.shape[0])).tocsr()
    mean_b_csr = sparse.coo_matrix(
        (mean, (row, col)), shape=(bias.shape[0], bias.shape[0])).tocsr()
    for i, cluster in enumerate(clusters):
        if classes[i] == 'up A':
            perturb_cluster(mean_a_csr, cluster, beta)
        elif classes[i] == 'down A':
            perturb_cluster(mean_a_csr, cluster, -beta)
        elif classes[i] == 'up B':
            perturb_cluster(mean_b_csr, cluster, beta)
        elif classes[i] == 'down B':
            perturb_cluster(mean_b_csr, cluster, -beta)

    # convert to COO
    mean_a_coo = mean_a_csr.tocoo()
    mean_b_coo = mean_b_csr.tocoo()
    del mean_a_csr
    del mean_b_csr

    # confirm index alignment
    assert np.all(mean_a_coo.row == row)
    assert np.all(mean_b_coo.row == row)
    assert np.all(mean_a_coo.col == col)
    assert np.all(mean_b_coo.col == col)
    assert np.all(mean_a_coo.data > 0)
    assert np.all(mean_b_coo.data > 0)

    eprint('  renaming cluster classes', skip=not verbose)
    classes[(classes == 'up A') | (classes == 'down B')] = 'A'
    classes[(classes == 'up B') | (classes == 'down A')] = 'B'

    eprint('  preparing generator', skip=not verbose)
    n_sim = size_factors.shape[-1]
    n_sim_per_cond = int(n_sim / 2)
    mean_a = mean_a_coo.data
    mean_b = mean_b_coo.data

    def gen():
        for j, m in zip(range(n_sim),
                        [mean_a]*n_sim_per_cond + [mean_b]*n_sim_per_cond):
            eprint('  biasing and simulating rep %i/%i' % (j+1, n_sim),
                   skip=not verbose)
            # compute aggregate bias factor
            if len(size_factors.shape) == 1:
                f = bias[row, j] * bias[col, j] * size_factors[j]
            else:
                f = bias[row, j] * bias[col, j] * size_factors[col-row, j]
            assert np.all(f > 0)
            assert np.all((m * f) > 0)

            # compute biased mean
            bm = m * f

            # establish cov
            cov = bm if trend == 'mean' else col - row

            # simulate
            yield sparse.coo_matrix(
                (freeze_distribution(
                    stats.nbinom, bm, mvr(bm, disp_fn(cov))).rvs(),
                 (row, col)), shape=(bias.shape[0], bias.shape[0]))\
                .tocsr()

    return classes, gen()
