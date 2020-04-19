Distance-conditional median of ratios demo
==========================================

In an interactive shell, import
:py:func:`hic3defdr.util.scaling.conditional_mor()`:

    >>> import numpy as np
    >>> from hic3defdr.util.scaling import conditional_mor

Create a test dataset with 4 replicates (columns) and 5 pixels (rows):

    >>> data = np.arange(20, dtype=float).reshape((5, 4))
    >>> data
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.],
           [16., 17., 18., 19.]])

Specify a distance for each pixel:

    >>> dist = np.array([1, 1, 1, 2, 2])


Normalize the data:

    >>> conditional_mor(data, dist)
    array([[0.79394639, 0.93946738, 1.08498836, 1.23050934],
           [0.79394639, 0.93946738, 1.08498836, 1.23050934],
           [0.79394639, 0.93946738, 1.08498836, 1.23050934],
           [0.90390183, 0.96968472, 1.0354676 , 1.10125049],
           [0.90390183, 0.96968472, 1.0354676 , 1.10125049]])
