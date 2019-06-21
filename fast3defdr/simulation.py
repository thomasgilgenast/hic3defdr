import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.ndimage as ndimage

from lib5c.util.distributions import freeze_distribution

from fast3defdr.scaled_nb import mvr


def perturb_loop(matrix, cluster, effect):
    """https://colab.research.google.com/drive/1dk9kX57ZtlxQ3jubrKL_q2r8LZnSlVwY"""
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
    matrix[r_slice, c_slice] += matrix[r_slice, c_slice].toarray() * \
        footprint * effect


def simulate(row, col, mean, disp_fn, bias, size_factors, clusters, beta=0.5,
             p_diff=0.4):
    print('  assigning cluster classes')
    classes = np.random.choice(
        np.array(['constit', 'up A', 'down A', 'up B', 'down B'], dtype='|S7'),
        size=len(clusters),
        p=[1 - p_diff, p_diff/4, p_diff/4, p_diff/4, p_diff/4]
    )

    print('  adding loops')
    mean_a_lil = sparse.coo_matrix(
        (mean, (row, col)), shape=(bias.shape[0], bias.shape[0])).tolil()
    mean_b_lil = sparse.coo_matrix(
        (mean, (row, col)), shape=(bias.shape[0], bias.shape[0])).tolil()
    del row
    del col
    for i, cluster in enumerate(clusters):
        if classes[i] == 'up A':
            perturb_loop(mean_a_lil, cluster, beta)
        elif classes[i] == 'down A':
            perturb_loop(mean_a_lil, cluster, -beta)
        elif classes[i] == 'up B':
            perturb_loop(mean_b_lil, cluster, beta)
        elif classes[i] == 'down B':
            perturb_loop(mean_b_lil, cluster, -beta)

    mean_a_csr = mean_a_lil.tocsr()
    mean_b_csr = mean_b_lil.tocsr()
    assert np.all(mean_a_csr.tocoo().data >= 0)
    assert np.all(mean_b_csr.tocoo().data >= 0)
    del mean_a_lil
    del mean_b_lil

    print('  computing new row and col index')
    mean_csr_sum_coo = (mean_a_csr + mean_b_csr).tocoo()
    new_row = mean_csr_sum_coo.row
    new_col = mean_csr_sum_coo.col
    del mean_csr_sum_coo

    print('  biasing mean matrices')
    n_sim = len(size_factors)
    mean_a = mean_a_csr[new_row, new_col].A1
    mean_b = mean_b_csr[new_row, new_col].A1
    assert np.all(mean_a >= 0)
    assert np.all(mean_b >= 0)
    biased_means = [mean*bias[new_row, i]*bias[new_col, i]*size_factors[i] + 0.1
                    for i, mean in zip(range(n_sim),
                                       [mean_a]*(n_sim/2) + [mean_b]*(n_sim/2))]
    assert all(np.all(m > 0) for m in biased_means)
    del mean_a
    del mean_b

    print('  simulating')
    for m in biased_means:
        yield sparse.coo_matrix(
            (freeze_distribution(stats.nbinom, m, mvr(m, disp_fn(m))).rvs(),
             (new_row, new_col)), shape=(bias.shape[0], bias.shape[0])).tocsr()
