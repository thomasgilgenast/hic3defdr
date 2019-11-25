from hic3defdr.analysis import HiC3DeFDR
from hic3defdr.plotting.roc import plot_roc
from hic3defdr.plotting.fdr import plot_fdr
from hic3defdr.plotting.fn_vs_fp import plot_fn_vs_fp
from hic3defdr.plotting.distance_bias import plot_distance_bias
from hic3defdr.plotting.dispersion import compare_disp_fits

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = [
    'HiC3DeFDR',
    'plot_roc',
    'plot_fdr',
    'plot_fn_vs_fp',
    'plot_distance_bias',
    'compare_disp_fits'
]
