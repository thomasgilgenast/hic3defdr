def context():
    """
    Infers the context in which the current program is being executed.

    https://stackoverflow.com/a/47428575

    Returns
    -------
    str
        The context, either 'colab', 'jupyter', 'ipython', or 'terminal'.
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'colab' in ipy_str:
            return 'colab'
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except NameError:
        return 'terminal'


try:
    if context() in ['jupyter', 'colab']:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
    tqdm_avail = True
    import tqdm as tqdm_module
except ImportError:
    tqdm_avail = False
    tqdm_module = None
    tqdm = None


def tqdm_maybe(iter, **kwargs):
    """
    Drop-in replacement from ``tqdm.tqdm()`` except will simply do nothing if
    ``tqdm`` is not present and will use ``tqdm.tqdm_notebook()`` if run in a
    notebook.

    Parameters
    ----------
    iter : iterable
        The iterable to wrap.
    kwargs : kwargs
        Will be passed through to ``tqdm.tqdm()``.

    Returns
    -------
    iterator
        Wrapped by ``tqdm`` if it was installed, or just ``iter`` otherwise.
    """
    if tqdm_avail:
        return tqdm(iter, **kwargs)
    return iter
