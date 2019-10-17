from multiprocessing import Pool, cpu_count

import dill


def _unpack_for_map(payload):
    fn, kwargs = dill.loads(payload)
    return fn(**kwargs)


def _pack_for_map(fn, kwargs_list):
    return [dill.dumps((fn, kwargs)) for kwargs in kwargs_list]


def parallel(fn, kwargs_list, n_threads=None):
    """
    Applies a function in parallel over a list of kwarg dicts, using a
    specified number of threads.

    Parameters
    ----------
    fn : function
        The function to parallelize.
    kwargs_list : list of dict
        The function will be called ``len(kwargs_list)`` times. Each time it is
        called it will use a different element of this list to determine the
        kwargs the function will be called with.
    n_threads : int
        Specify the number of subprocesses to use for parallelization. Pass -1
        to use as many subprocesses as there are CPUs.
    """
    if n_threads == -1:
        n_threads = cpu_count()
    pool = Pool(n_threads)
    pool.map(_unpack_for_map, _pack_for_map(fn, kwargs_list))
