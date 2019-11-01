from multiprocessing import Pool, cpu_count

import dill

from hic3defdr.progress import tqdm_avail, tqdm


def _pack_for_apply(fn, kwargs_list):
    return [dill.dumps((fn, kwargs)) for kwargs in kwargs_list]


def _unpack_for_apply(payload):
    fn, kwargs = dill.loads(payload)
    return fn(**kwargs)


def parallel_apply(fn, kwargs_list, n_threads=None):
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
    result = pool.map(_unpack_for_apply, _pack_for_apply(fn, kwargs_list))
    pool.close()
    pool.join()


def _pack_for_map(fn, kwargs):
    return dill.dumps((fn, kwargs))


def _unpack_for_map(payload):
    fn, kwargs = dill.loads(payload)
    idx = kwargs['_idx']
    del kwargs['_idx']
    return idx, fn(**kwargs)


def parallel_map(fn, kwargs_list, n_threads=None):
    """
    Maps a function in parallel over a list of kwarg dicts, using a specified
    number of threads.

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
    # resolve n_threads
    if n_threads == -1:
        n_threads = cpu_count()

    # create a progress bar and result list
    n = len(kwargs_list)
    pbar = tqdm(total=n) if tqdm_avail else None
    result = [None] * n

    # fill in magic kwarg _idx
    for i in range(n):
        kwargs_list[i]['_idx'] = i

    # callback for apply_async
    # payload is the return value of _unpack_for_map
    def update(payload):
        i, r = payload
        result[i] = r
        if tqdm_avail:
            pbar.update()

    # process in parallel
    pool = Pool(n_threads)
    for kwargs in kwargs_list:
        pool.apply_async(_unpack_for_map, args=(_pack_for_map(fn, kwargs),),
                         callback=update)
    pool.close()
    pool.join()
    if pbar is not None:
        pbar.close()
    return result
