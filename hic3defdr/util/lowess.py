import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from lib5c.util.lowess import lowess

from hic3defdr.util.printing import eprint


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
                        right_boundary=None, frac=None, auto_frac_factor=15.,
                        delta=0.01, w=20, power=1./4,
                        interpolate_before_increase=True):
    """
    Performs lowess fitting as in ``lowess_fit()``, but weighting the data
    points automatically according to the precision in the ``y`` values as
    estimated by a rolling window sample variance.

    Points are weighted proportionally to a specified power ``power`` of their
    precision by adding duplicated points to the dataset. This should
    approximate the effects of a true weighted lowess fit, with the caveat that
    the weights are rounded a bit.

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
    frac : float, optional
        The lowess smoothing fraction to use. Pass None to use the default:
        ``auto_frac_factor`` divided by the product of the average of the
        unscaled weights and the largest scaled weight.
    auto_frac_factor : float
        When ``frac`` is None, this factor scales the automatically determined
        fraction parameter.
    delta : float
        Distance (on the scale of ``x`` or ``log(x)``) within which to use
        linear interpolation when constructing the initial fit, expressed as a
        fraction of the range of ``x`` or ``log(x)``.
    w : int
        The size of the rolling window to use when estimating the precision of
        the y values.
    power : float
        Precisions will be taken to this power to obtain unscaled weights.
    interpolate_before_increase : bool
        Hacky flag introduced to handle quirk of Hi-C dispersion vs distance
        relationships in which dispersion is elevated at extremely short
        distances. When True, this function will identify a group of points with
        the lowest x-values across which the y-value is monotonically
        decreasing. These points will be included in the variance estimation,
        but will be excluded from lowess fitting. Linear interpolation will be
        used at these x-values instead, since it is hard to convince lowess to
        follow a sharp change in the trend that is only supported by 3-4 data
        points out of 200-500 total data points, even with our best attempts at
        weighting. Pass False to perform a simple weighted lowess fit with no
        linear interpolation.

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
    sort_idx = np.argsort(x)
    x = x[sort_idx].copy()
    y = y[sort_idx].copy()

    # compute rolling var
    var = pd.Series(y).rolling(window=w, center=True).var().values

    # convert to precision
    prec = 1 / var

    # apply power to obtain weights
    weight = np.ones_like(var) * np.nan
    weight[np.isfinite(prec)] = np.power(prec[np.isfinite(prec)], power)

    # scale to make smallest weight 1
    min_weight = np.nanmin(weight)
    scaled_weight = weight * (1 / min_weight)

    # clip "infinite weight" points to the max weight
    max_weight = np.nanmax(scaled_weight)
    scaled_weight[np.isinf(scaled_weight)] = max_weight

    # fill left and right side nan's
    # get the first non-nan precision (for filling left side)
    left_weight = scaled_weight[np.argmax(np.isfinite(scaled_weight))]
    left_fill_idx = np.isnan(scaled_weight) & (i < n/2)
    right_fill_idx = np.isnan(scaled_weight) & (i > n/2)
    scaled_weight[left_fill_idx] = left_weight
    # fill right side with 1s
    scaled_weight[right_fill_idx] = 1
    assert np.all(np.isfinite(scaled_weight))

    # floor and convert to int
    floored_weight = np.floor(scaled_weight).astype(int)

    # find index of first increase
    inc_idx = np.argmax(np.diff(y) > 0) + 1 if interpolate_before_increase \
        else 0

    # create duplicated data
    expanded_xs = []
    expanded_ys = []
    for i in range(inc_idx, n):
        m = floored_weight[i]
        if not np.isfinite(m):
            continue
        expanded_xs.extend([x[i]] * m)
        expanded_ys.extend([y[i]] * m)

    # resolve frac
    if frac is None:
        frac_auto = auto_frac_factor / (max_weight * np.nanmean(weight))
        frac = max(min(frac_auto, 2./3), 0.05)
        eprint('  using auto-determined lowess fraction of %.3f' % frac)

    # lowess fit duplicated data
    lowess_fn = lowess_fit(
        np.array(expanded_xs), np.array(expanded_ys), logx=logx, logy=logy,
        left_boundary=left_boundary, right_boundary=right_boundary, frac=frac,
        delta=delta)

    def fit(x_star):
        x_star = np.asarray(x_star)
        interp_y_hat = interp1d(x, y, bounds_error=False,
                                fill_value='extrapolate')(x_star)
        interp_y_hat[x_star < x[0]] = y[0]
        fit_y_hat = lowess_fn(x_star)
        interp_idx = x_star < x[inc_idx]
        if type(interp_idx) == bool:
            if interp_idx:
                return interp_y_hat
            else:
                return fit_y_hat
        fit_y_hat[interp_idx] = interp_y_hat[interp_idx]
        return fit_y_hat

    return fit
