from __future__ import print_function
import sys


def eprint(*args, **kwargs):
    if not kwargs.get('skip', False):
        if 'skip' in kwargs:
            del kwargs['skip']
        print(*args, file=sys.stderr, **kwargs)
