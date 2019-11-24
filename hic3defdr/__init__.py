from hic3defdr.analysis import HiC3DeFDR
from hic3defdr.plotting.roc import plot_roc
from hic3defdr.plotting.fdr import plot_fdr
from hic3defdr.plotting.dispersion import compare_disp_fits

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = [
    'HiC3DeFDR',
    'plot_roc',
    'plot_fdr',
    'compare_disp_fits'
]
