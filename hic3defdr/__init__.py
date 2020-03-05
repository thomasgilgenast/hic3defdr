try:
    from hic3defdr.analysis import HiC3DeFDR
    from hic3defdr.plotting.roc import plot_roc
    from hic3defdr.plotting.fdr import plot_fdr
    from hic3defdr.plotting.fn_vs_fp import plot_fn_vs_fp
    from hic3defdr.plotting.distance_bias import plot_distance_bias
    from hic3defdr.plotting.dispersion import compare_disp_fits

    __all__ = [
        'HiC3DeFDR',
        'plot_roc',
        'plot_fdr',
        'plot_fn_vs_fp',
        'plot_distance_bias',
        'compare_disp_fits'
    ]
except ImportError:
    print('WARNING: hic3defdr has not been installed successfully yet')

try:
    try:
        # this works in Python 3.8
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        try:
            # this works in Python 2 if hic3defdr is installed, since it depends
            # on importlib_metadata
            from importlib_metadata import version, PackageNotFoundError
        except ImportError:
            raise
    try:
        # we land here if either importlib.metadata or importlib_metadata
        # is available and hic3defdr is installed
        __version__ = version(__name__)
    except PackageNotFoundError:
        # we will land here if either importlib.metadata or importlib_metadata
        # is available, but hic3defdr isn't actually installed
        __version__ = 'unknown'
except ImportError:
    # we land here if neither importlib.metadata nor importlib_metadata are
    # available
    __version__ = 'unknown'
