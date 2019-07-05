import numpy as np
from scipy.interpolate import interp1d

from lib5c.util.lowess import lowess


def lowess_fit(x, y, logx=False, logy=False, left_boundary=None,
               right_boundary=None, frac=0.3, delta=0.01):
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
