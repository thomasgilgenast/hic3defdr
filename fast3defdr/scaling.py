import numpy as np

from lib5c.util.mathematics import gmean


def median_of_ratios(data):
    return np.median(data / gmean(data, axis=1)[:, None], axis=0)
