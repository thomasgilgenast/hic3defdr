def context():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except NameError:
        return 'terminal'


try:
    if context() == 'jupyter':
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
    tqdm_avail = True
except ImportError:
    tqdm_avail = False
    tqdm = None


def tqdm_maybe(iter, **kwargs):
    if tqdm_avail:
        return tqdm(iter, **kwargs)
    return iter
