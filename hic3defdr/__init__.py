from hic3defdr.analysis import HiC3DeFDR
from hic3defdr.plotting.roc import plot_roc
from hic3defdr.plotting.fdr import plot_fdr

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = [
    'HiC3DeFDR',
    'plot_roc',
    'plot_fdr',
]
