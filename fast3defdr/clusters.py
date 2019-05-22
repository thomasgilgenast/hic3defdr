import json
from itertools import izip

import numpy as np
import scipy.sparse as sparse
from scipy.ndimage import generate_binary_structure

from lib5c.util.system import check_outdir


class DirectedDisjointSet(object):
    """
    Based on https://stackoverflow.com/a/3067672 but with two colors.

    The overall effect is like a directed sparse graph - DDS.add(a, b) is like
    adding an edge from a to b. a gets marked as a source, b does not (anything
    not in the set DDS.sources is assumed to be a destination). If b is in an
    existing group, but isn't also the source of any other edge, then the groups
    won't be merged. Finally, RBDS.get_groups() will be filtered to include only
    source nodes.

    This is an improved version where destination nodes are not stored anywhere
    if they haven't previously been seen as a source.
    """
    def __init__(self):
        self.leader = {}  # maps a member to the group's leader
        self.group = {}  # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                # we've seen both members before, need to merge the groups
                if leadera == leaderb:
                    return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = \
                        b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                # a has been seen before but b doesn't exist - do nothing
                return
        else:
            if leaderb is not None:
                # a hasn't been seen before but b exists - connect them
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                # a hasn't been seen before and b doesn't exist
                # create a new group for a, ignore b
                self.leader[a] = a
                self.group[a] = {a}

    def get_groups(self):
        return self.group.values()


def find_clusters(sig_points, connectivity=1):
    spmatrix = sparse.coo_matrix(sig_points)
    structure = generate_binary_structure(2, connectivity)
    xs, ys = np.where(structure)
    xs -= 1
    ys -= 1
    shifts = [np.array(c) for c in zip(xs, ys)]
    dds = DirectedDisjointSet()
    for c in izip(spmatrix.row, spmatrix.col):
        center = np.array(c)
        for shift in shifts:
            dds.add(tuple(center), tuple(center + shift))
    return dds.get_groups()


class NumpyEncoder(json.JSONEncoder):
    """
    Pass this to `json.dump()` to correctly serialize numpy values.

    Credit: https://stackoverflow.com/a/27050186
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def save_clusters(clusters, outfile):
    """
    Saves cluster information to disk in sparse JSON format.

    Parameters
    ----------
    clusters : np.ndarray or list of set of tuple of int
        If an `np.ndarray` is passed, it should be square and triangular and
        have int dtype. Entries should be the cluster id for points which belong
        to that cluster, zero everywhere else. If a list of sets is passed, the
        sets are clusters, the tuples are the indices of entries in that
        cluster.
    outfile : str
        File to write JSON output to.
    """
    check_outdir(outfile)
    with open(outfile, 'w') as handle:
        if type(clusters) == np.ndarray:
            clusters = convert_cluster_array_to_sparse(clusters)
        json.dump([[[i, j] for i, j in c] for c in clusters], handle,
                  cls=NumpyEncoder)


def convert_cluster_array_to_sparse(cluster_array):
    """
    Converts an array of cluster information to a sparse, JSON-friendly format.

    Parameters
    ----------
    cluster_array : np.ndarray or scipy.sparse.spmatrix
        Square, triangular, int dtype. Entries should be the cluster id for
        points which belong to that cluster, zero everywhere else.

    Returns
    -------
    list of sets of tuples of int
        The sets are clusters, tuples are the matrix indices of the pixels in
        that cluster.

    Notes
    -----
    Since the introduction of `hiclite.util.clusters.find_clusters()`, this
    function is no longer used.
    """
    obj = {}
    if isinstance(cluster_array, sparse.coo_matrix):
        x = cluster_array
    elif isinstance(cluster_array, sparse.spmatrix):
        x = cluster_array.tocoo()
    else:
        x = sparse.coo_matrix(cluster_array)
    for i, j, idx in izip(x.row, x.col, x.data):
        if not idx:
            continue
        if idx not in obj:
            obj[int(idx)] = set()
        obj[int(idx)].add((int(i), int(j)))
    return obj.values()


def load_clusters(infile):
    """
    Loads clusters in a sparse format from a JSON file.

    Parameters
    ----------
    infile : str
        The JSON file containing sparse cluster information.

    Returns
    -------
    list of set of tuple of int
        The sets are clusters, the tuples are the indices of entries in that
        cluster.
    """
    with open(infile, 'r') as handle:
        return [set([tuple(e) for e in cluster])
                for cluster in json.load(handle)]


def clusters_to_coo(clusters, shape):
    """
    Converts clusters (list of list of tuple) to a COO sparse matrix.

    Parameters
    ----------
    clusters : list of list of tuple
        The outer list is a list of clusters. Each cluster is a list of (i, j)
        tuples marking the position of significant points which belong to that
        cluster.
    shape : tuple
        The shape with which to construct the COO matrix.

    Returns
    -------
    scipy.sparse.coo_matrix
        The sparse matrix of significant points.

    Examples
    --------
    >>> from fast3defdr.clusters import clusters_to_coo
    >>> coo = clusters_to_coo([[(1, 2), (1, 1)], [(4, 4),  (3, 4)]], (5, 5))
    >>> coo.toarray()
    array([[False, False, False, False, False],
           [False,  True,  True, False, False],
           [False, False, False, False, False],
           [False, False, False, False,  True],
           [False, False, False, False,  True]])
    """
    if not clusters:
        return sparse.coo_matrix(shape, dtype=bool)
    i, j = zip(*[loop for cluster in clusters for loop in cluster])
    return sparse.coo_matrix((np.ones(len(i), dtype=bool), (i, j)), shape=shape)
