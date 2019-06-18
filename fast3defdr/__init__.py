from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from fast3defdr.analysis import Fast3DeFDR
