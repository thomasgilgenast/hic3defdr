import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from lib5c.util.lowess import lowess


def lowess_fit(x, y, logx=False, logy=False, left_boundary=None,
               right_boundary=None, frac=0.3, delta=0.01):
    """
    Opinionated convenience wrapper for lowess smoothing.

    Parameters
    ----------
    x, y : np.ndarray
        The x and y values to fit, respectively.
    logx, logy : bool
        Pass True to perform the fit on the scale of ``log(x)`` and/or
        ``log(y)``, respectively.
    left_boundary, right_boundary : float, optional
        Allows specifying boundaries for the fit, in the original ``x`` space.
        If a float is passed, the returned fit will return the farthest left or
        farthest right lowess-estimated ``y_hat`` (from the original fitting
        set) for all points which are left or right of the specified left or
        right boundary point, respectively. Pass None to use linear
        extrapolation for these points instead.
    frac : float
        The lowess smoothing fraction to use.
    delta : float
        Distance (on the scale of ``x`` or ``log(x)``) within which to use
        linear interpolation when constructing the initial fit, expressed as a
        fraction of the range of ``x`` or ``log(x)``.

    Returns
    -------
    function
        This function takes in ``x`` values on the original ``x`` scale and
        returns estimated ``y`` values on the original ``y`` scale (regardless
        of what is passed for ``logx`` and ``logy``). This function will still
        return sane estimates for ``y`` even at points not in the original
        fitting set by performing linear interpolation in the space the fit was
        performed in.

    Notes
    -----
    No filtering of input values is performed; clients are expected to handle
    this if desired. NaN values should not break the function, but ``x`` points
    with zero values passed when ``logx`` is True are expected to break the
    function.

    The default value of the ``delta`` parameter is set to be non-zero, matching
    the behavior of lowess smoothing in R and improving performance.

    Linear interpolation between x-values in the original fitting set is used to
    provide a familiar functional interface to the fitted function.

    Boundary conditions on the fitted function are exposed via ``left_boundary``
    and ``right_boundary``, mostly as a convenience for points where ``x == 0``
    when fitting was performed on the scale of ``log(x)``.

    When ``left_boundary`` or ``right_boundary`` are None (this is the default)
    the fitted function will be linearly extrapolated for points beyond the
    lowest and highest x-values in ``x``.
    """
    if logx:
        x = np.log(x)
    if logy:
        y = np.log(y)

    res = lowess(y, x, frac=frac, delta=(np.nanmax(x) - np.nanmin(x)) * delta)
    sorted_x = res[:, 0]
    sorted_y_hat = res[:, 1]

    def fit(x_star):
        if logx:
            new_x = np.log(x_star)
        else:
            new_x = x_star
        _, idx = np.unique(sorted_x, return_index=True)
        y_hat = interp1d(sorted_x[idx], sorted_y_hat[idx],
                         fill_value='extrapolate', assume_sorted=True)(new_x)
        if left_boundary is not None:
            y_hat[x_star <= left_boundary] = sorted_y_hat[0]
        if right_boundary is not None:
            y_hat[x_star >= right_boundary] = sorted_y_hat[-1]
        if logy:
            y_hat = np.exp(y_hat)
        return y_hat

    return fit


def weighted_lowess_fit(x, y, logx=False, logy=False, left_boundary=None,
                        right_boundary=None, frac=0.2, delta=0.01, w=20):
    """
    Performs lowess fitting as in ``lowess_fit()``, but weighting the data
    points automatically according to the precision in the ``y`` values as
    estimated by a rolling window sample variance.

    Points are weighted proportionally to their precision by adding duplicated
    points to the dataset. This should approximate the effects of a true
    weighted lowess fit, with the caveat that the weights are rounded a bit.

    Weighting the data points according to this rolling window sample variance
    is probably only a good idea if the marginal distribution of ``x`` values is
    uniform.

    Parameters
    ----------
    x, y : np.ndarray
        The x and y values to fit, respectively.
    logx, logy : bool
        Pass True to perform the fit on the scale of ``log(x)`` and/or
        ``log(y)``, respectively.
    left_boundary, right_boundary : float, optional
        Allows specifying boundaries for the fit, in the original ``x`` space.
        If a float is passed, the returned fit will return the farthest left or
        farthest right lowess-estimated ``y_hat`` (from the original fitting
        set) for all points which are left or right of the specified left or
        right boundary point, respectively. Pass None to use linear
        extrapolation for these points instead.
    frac : float
        The lowess smoothing fraction to use.
    delta : float
        Distance (on the scale of ``x`` or ``log(x)``) within which to use
        linear interpolation when constructing the initial fit, expressed as a
        fraction of the range of ``x`` or ``log(x)``.
    w : int
        The size of the rolling window to use when estimating the precision of
        the y values.

    Returns
    -------
    function
        This function takes in ``x`` values on the original ``x`` scale and
        returns estimated ``y`` values on the original ``y`` scale (regardless
        of what is passed for ``logx`` and ``logy``). This function will still
        return sane estimates for ``y`` even at points not in the original
        fitting set by performing linear interpolation in the space the fit was
        performed in.
    """
    n = len(y)
    i = np.arange(n)

    # compute rolling var
    var = pd.Series(y).rolling(window=w, center=True).var().values

    # convert to precision
    prec = 1 / var

    # scale to make smallest precision 1
    min_prec = np.nanmin(prec)
    scaled_prec = prec * (1 / min_prec)

    # fill left and right side nan's
    # get the first non-nan precision (for filling left side)
    max_prec = scaled_prec[np.argmax(np.isfinite(scaled_prec))]
    max_fill_idx = np.isnan(scaled_prec) & (i < n/2)
    min_fill_idx = np.isnan(scaled_prec) & (i > n/2)
    scaled_prec[max_fill_idx] = max_prec
    # fill right side with 1s
    scaled_prec[min_fill_idx] = 1

    # floor and convert to int
    floored_prec = np.floor(scaled_prec).astype(int)

    # create duplicated data
    expanded_xs = []
    expanded_ys = []
    for i in range(n):
        m = floored_prec[i]
        if not np.isfinite(m):
            continue
        expanded_xs.extend([x[i]] * m)
        expanded_ys.extend([y[i]] * m)

    # lowess fit duplicated data
    return lowess_fit(
        np.array(expanded_xs), np.array(expanded_ys), logx=logx, logy=logy,
        left_boundary=left_boundary, right_boundary=right_boundary, frac=frac,
        delta=delta)
