import numpy as np
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
        y_hat = interp1d(sorted_x, sorted_y_hat, fill_value='extrapolate',
                         assume_sorted=True)(new_x)
        if left_boundary is not None:
            y_hat[x_star <= left_boundary] = sorted_y_hat[0]
        if right_boundary is not None:
            y_hat[x_star >= right_boundary] = sorted_y_hat[-1]
        if logy:
            y_hat = np.exp(y_hat)
        return y_hat

    return fit
