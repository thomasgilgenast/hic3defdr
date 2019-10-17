from __future__ import print_function
import sys


def eprint(*args, **kwargs):
    """
    Drop-in replacement for ``print()`` that prints to stderr instead of stdout.

    Parameters
    ----------
    args, kwargs : args, kwargs
        All args and kwargs will be passed through to ``print()`` except for the
        special kwarg ``skip``; if this kwarg is present and set to True,
        nothing will be printed.
    """
    if not kwargs.get('skip', False):
        if 'skip' in kwargs:
            del kwargs['skip']
        print(*args, file=sys.stderr, **kwargs)
