from multiprocessing import Pool, cpu_count

import dill


def _unpack_for_map(payload):
    fn, kwargs = dill.loads(payload)
    return fn(**kwargs)


def _pack_for_map(fn, kwargs_list):
    return [dill.dumps((fn, kwargs)) for kwargs in kwargs_list]


def parallel(fn, kwargs_list, n_threads=None):
    if n_threads == -1:
        n_threads = cpu_count()
    pool = Pool(n_threads)
    pool.map(_unpack_for_map, _pack_for_map(fn, kwargs_list))
