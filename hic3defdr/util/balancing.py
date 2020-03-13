import numpy as np
import scipy.sparse as sparse


def kr_balance(array, tol=1e-6, x0=None, delta=0.1, ddelta=3, fl=1,
               max_iter=3000):
    """
    Balances a matrix via Knight-Ruiz matrix balancing, using sparse matrix
    operations.

    Parameters
    ----------
    array : np.ndarray or scipy.sparse.spmatrix
        The matrix to balance. Must be 2 dimensional and square. Will be
        symmetrized using its upper triangular entries.
    tol : float
        The error tolerance.
    x0 : np.ndarray, optional
        The initial guess for the bias vector. Pass None to use a vector of all
        1's.
    delta, ddelta : float
        How close/far the balancing vectors can get to/from the positive cone,
        in terms of a relative measure on the size of the elements.
    fl : int
        Determines whether or not internal convergence statistics will be
        printed to standard output.
    max_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    sparse.csr_matrix, np.ndarray, np.ndarray
        The CSR matrix is the balanced matrix. It will be upper triangular if
        ``array`` was upper triangular, otherwise it will by symmetric. The
        first np.ndarray is the bias vector. The second np.ndarray is a list of
        residuals after each iteration.

    Notes
    -----
    The core logic of this implementation is transcribed (by Dan Gillis) from
    the original Knight and Ruiz IMA J. Numer. Anal. 2013 paper and differs only
    slightly from the implementation in Juicer/Juicebox (see comments).

    It then uses the "sum factor" approach from the Juicer/Juicebox code to
    scale the bias vectors up to match the scale of the original matrix (see
    comments).

    Overall, this function is not nan-safe, but you may pass matrices that
    contain empty rows (the matrix will be shrunken before balancing, but all
    outputs will match the shape of the original matrix).

    This function does not perform any of the logic used in Juicer/Juicebox to
    filter out sparse rows if the balancing fails to converge.

    If the max_iter limit is reached, this function will return the current best
    balanced matrix and bias vector - no exception will be thrown. Callers can
    compare the length of the list of residuals to max_iter - if they are equal,
    the algorithm has not actually converged, and the caller may choose to
    perform further filtering on the matrix and try again. The caller is also
    free to decide if the matrix is "balanced enough" using any other criteria
    (e.g., variance of nonzero row sums).
    """
    # check to see if input is upper triangular
    triu = sparse.tril(array, k=-1).nnz == 0

    # convert to CSR and symmetrize
    array = sparse.triu(sparse.csr_matrix(array))
    array += array.transpose() - sparse.diags([array.diagonal()], [0])

    # shrink matrix, storing copy of original full matrix
    nonzero = array.getnnz(1) > 0
    full_array = array.copy()
    array = array[nonzero, :][:, nonzero]

    # this block was copied from lib5c.algorithms.knight_ruiz.kr_balance() which
    # was transcribed by Dan Gillis from the Matlab source provided in the
    # supplement of Knight and Ruiz IMA J. Numer. Anal. 2013
    # the only changes made in this block (relative to the lib5c version) are
    # that the dot products have been rewritten to work with sparse matrices
    # the Matlab/Python implementation is nearly equivalent to the Java
    # function compteKRNormVector() at
    # https://github.com/theaidenlab/Juicebox/blob/
    # 5a56089c63957cb15401ea7906ab77e242dfd755/src/juicebox/tools/utils/
    # original/NormalizationCalculations.java#L138
    # except that
    # 1) the Matlab/Python implementation has a concept of Delta/ddelta, which
    #    serves as an upper limit analogue to delta
    # 2) the Java version stops infinite iteration using a not_changing counter,
    #    which terminates the iteration when rho_km1 hasn't changed
    #    significantly in 100 iterations and returns a failing result; the
    #    Python version uses a max_iter counter, which simply sets a maximum
    #    number of iterations to perform but does not return a failing result if
    #    the maximum number of iterations is exceeded; the Matlab implementation
    #    does neither and will instead loop forever
    dims = array.shape
    if dims[0] != dims[1] or len(dims) > 2:
        raise Exception
    n = dims[0]
    e = np.ones((n, 1))
    res = np.array([])
    if x0 is None:
        x0 = e.copy()
    g = np.float_(0.9)
    eta_max = 0.1
    eta = eta_max
    stop_tol = tol * 0.5
    # x0 is not used again, so do not need to copy object
    x = x0  # n x 1 array
    rt = tol ** 2
    v = x0 * array.dot(x)  # n x 1 array
    rk = 1 - v  # n x 1 array
    rho_km1 = np.dot(np.transpose(rk), rk)[0, 0]  # scalar value
    # no need to copy next two objects - are scalar values
    rout = rho_km1
    rold = rout
    mvp = 0
    i = 0
    if fl == 1:
        print('it in. it res')
    while rout > rt:
        if max_iter is not None and i > max_iter:
            break
        i += 1
        k = 0
        y = np.ones((n, 1))
        innertol = max(eta ** 2 * rout, rt)
        rho_km2 = None
        while rho_km1 > innertol:
            k += 1
            if k == 1:
                z = rk / v
                p = z.copy()
                rho_km1 = np.dot(np.transpose(rk), z)
            else:
                beta = rho_km1 / rho_km2
                p = z + beta * p
            w = x * array.dot(x * p) + v * p
            alpha = rho_km1 / np.dot(np.transpose(p), w)
            ap = alpha * p
            ynew = y + ap
            if np.min(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0)
                gamma = np.min((delta - y[ind]) / ap[ind])
                y = y + gamma * ap
                break
            if np.max(ynew) >= ddelta:
                # see above - need to code
                # also check that (i.e. gamma or ap is scalar)
                ind = np.where(ynew > ddelta)
                gamma = min((ddelta - y[ind]) / ap[ind])
                y = y + gamma * ap
                break
            y = ynew
            rk = rk - alpha * w
            rho_km2 = rho_km1
            z = rk / v
            rho_km1 = np.dot(np.transpose(rk), z)
        x = x * y
        v = x * array.dot(x)
        rk = 1 - v
        rho_km1 = np.dot(np.transpose(rk), rk)[0, 0]  # scalar value
        rout = rho_km1
        mvp = mvp + k + 1
        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_0 = eta
        eta = g * rat
        if g * eta_0 ** 2 > 0.1:
            eta = max(eta, g * eta_0 ** 2)
        eta = max(min(eta, eta_max), stop_tol / res_norm)
        if fl == 1:
            print('{} {} {:.3e}'.format(i, k, res_norm))
            res = np.append(res, res_norm)
    del array

    # expand bias vector
    bias = np.zeros_like(nonzero, dtype=float)
    bias[nonzero] = np.squeeze(x)

    # compute balanced matrix
    bias_csr = sparse.diags([bias], [0])
    balanced = bias_csr.dot(full_array)
    balanced = balanced.dot(bias_csr)

    # compute "sum factor" and recompute balanced matrix
    # this is equivalent to the Java function getSumFactor() at
    # https://github.com/theaidenlab/Juicebox/blob/
    # 5a56089c63957cb15401ea7906ab77e242dfd755/src/juicebox/tools/utils/
    # original/NormalizationCalculations.java#L333
    sum_factor = np.sqrt(full_array.sum() / balanced.sum())
    bias *= sum_factor
    bias_csr = sparse.diags([bias], [0])
    balanced = bias_csr.dot(full_array)
    balanced = balanced.dot(bias_csr)
    del full_array

    # invert bias vector at nonzero positions
    bias[bias != 0] = 1 / bias[bias != 0]

    # if input was upper triangular, return upper triangular balanced matrix
    if triu:
        balanced = sparse.triu(balanced)
        balanced = balanced.tocsr()

    return balanced, bias, res
