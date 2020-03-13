import os
from six.moves.urllib.request import urlretrieve
from zipfile import ZipFile

from hic3defdr.util.printing import eprint


DEMO_DATA_URL = 'https://www.dropbox.com/sh/mq0fpnp4jz59wpo/AAD2FW1Tp_mVKCkxlJoZvxC8a?dl=1'  # noqa: E501
CHROMS = ['chr18', 'chr19']
CONDS = ['ES', 'NPC']
COND_EXT = ['clusters.json']
REPS = ['ES_1', 'ES_3', 'NPC_1', 'NPC_2']
REP_EXT = ['kr.bias', 'raw.npz']


def check_demo_data(dest_dir='~/hic3defdr-demo-data'):
    """
    Checks to see if all demo files are present.

    Parameters
    ----------
    dest_dir : str
        Path to destination directory to look for demo data files in.

    Returns
    -------
    bool
        True if all demo files are present, False otherwise.
    """
    dest_dir = os.path.expanduser(dest_dir)
    for chrom in CHROMS:
        for cond in CONDS:
            for ext in COND_EXT:
                path = os.path.join(
                    dest_dir, 'clusters', '%s_%s_%s' % (cond, chrom, ext))
                if not os.path.exists(path):
                    eprint('failed to find expected demo file at %s' % path)
                    return False
        for rep in REPS:
            for ext in REP_EXT:
                path = os.path.join(dest_dir, rep, '%s_%s' % (chrom, ext))
                if not os.path.exists(path):
                    eprint('failed to find expected demo file at %s' % path)
                    return False
    return True


def ensure_demo_data(dest_dir='~/hic3defdr-demo-data'):
    """
    Checks that all the demo data files are present to dest_dir, and downloads
    them if any are missing.

    Parameters
    ----------
    dest_dir : str
        Path to destination directory to download files to.
    """
    dest_dir = os.path.expanduser(dest_dir)
    if check_demo_data(dest_dir=dest_dir):
        eprint('demo data is present on disk, not re-downloading')
        return
    eprint('downloading demo data to temp.zip')
    urlretrieve(DEMO_DATA_URL, 'temp.zip')
    eprint('extracting demo data to %s' % dest_dir)
    try:
        with ZipFile('temp.zip', 'r') as z:
            z.extractall(dest_dir)
    finally:
        eprint('deleting temp.zip')
        os.remove('temp.zip')


def main():
    ensure_demo_data()


if __name__ == '__main__':
    main()
