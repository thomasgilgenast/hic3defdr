import pandas as pd

from lib5c.util.bed import parse_feature_from_string
from lib5c.util.primers import natural_sort_key
from hic3defdr.util.clusters import cluster_to_loop_id, cluster_from_string


COLUMN_ORDER = ['loop_id', 'us_chrom', 'us_start', 'us_end', 'ds_chrom',
                'ds_start', 'ds_end', 'cluster_size', 'cluster']


def clusters_to_table(clusters, chrom, res):
    """
    Creates a DataFrame which tabulates cluster information.

    The DataFrame's first column (and index) will be a "loop_id" in the form
    "chr:start-end_chr:start-end". Its other columns will be "us_chrom",
    "us_start", "us_end", and "ds_chrom", "ds_start", "ds_end", representing the
    BED-style chromosome, start coordinate, and end coordinate of the upstream
    ("us", smaller coordinate values) and downstream ("ds", larger coordinate
    values) anchors of the loop, respectively. These anchors together form a
    rectangular "bounding box" that completely encloses the significant pixels
    in the cluster. The DataFrame will also have a "cluster_size" column
    representing the total number of significant pixels in the cluster. Finally,
    the exact indices of the significant pixels in the cluster will be recorded
    in a "cluster" column in a JSON-like format (using only square brackets).

    Parameters
    ----------
    clusters : list of list of tuple
        The outer list is a list of clusters. Each cluster is a list of (i, j)
        tuples marking the position of significant points which belong to that
        cluster.
    chrom : str
        The name of the chromosome these clusters are on.
    res : int
        The resolution of the contact matrix referred to by the row and column
        indices in ``clusters``, in units of base pairs.

    Returns
    -------
    pd.DataFrame
        The table of loop information.

    Examples
    --------
    >>> from hic3defdr.util.cluster_table import clusters_to_table
    >>> clusters = [[(1, 2), (1, 1)], [(4, 4),  (3, 4)]]
    >>> df = clusters_to_table(clusters, 'chrX', 10000)
    >>> df.iloc[0, :]
    us_chrom                    chrX
    us_start                   10000
    us_end                     20000
    ds_chrom                    chrX
    ds_start                   10000
    ds_end                     30000
    cluster_size                   2
    cluster         [[1, 2], [1, 1]]
    Name: chrX:10000-20000_chrX:10000-30000, dtype: object
    """
    loops = []
    for cluster in clusters:
        cluster = list(cluster)
        loop_id = cluster_to_loop_id(cluster, chrom, res)
        us_anchor, ds_anchor = map(
            parse_feature_from_string, loop_id.split('_'))
        loops.append({
            'loop_id': loop_id,
            'us_chrom': us_anchor['chrom'],
            'us_start': us_anchor['start'],
            'us_end': us_anchor['end'],
            'ds_chrom': ds_anchor['chrom'],
            'ds_start': ds_anchor['start'],
            'ds_end': ds_anchor['end'],
            'cluster_size': len(cluster),
            'cluster': [list(c) for c in cluster]
        })
    return sort_cluster_table(
        pd.DataFrame(loops, columns=COLUMN_ORDER).set_index('loop_id'))


def sort_cluster_table(cluster_table):
    r"""
    Sorts the rows of a cluster table in the expected order.

    This function does not operate in-place.

    We expect this to get a lot easier after this pandas issue is fixed:
    https://github.com/pandas-dev/pandas/issues/3942

    Parameters
    ----------
    cluster_table : pd.DataFrame
        The cluster table to sort. Must have all the expected columns.

    Returns
    -------
    pd.DataFrame
        The sorted cluster table.

    Examples
    --------
    >>> from hic3defdr.util.cluster_table import clusters_to_table, \
    ...     sort_cluster_table
    >>> clusters = [[(4, 4),  (3, 4)], [(1, 2), (1, 1)]]
    >>> res = 10000
    >>> df1 = clusters_to_table(clusters, 'chr1', res)
    >>> df2 = clusters_to_table(clusters, 'chr2', res)
    >>> df3 = clusters_to_table(clusters, 'chr11', res)
    >>> df4 = clusters_to_table(clusters, 'chrX', res)
    >>> df = pd.concat([df4, df3, df2, df1], axis=0)
    >>> sort_cluster_table(df).index
    Index([u'chr1:10000-20000_chr1:10000-30000',
           u'chr1:30000-50000_chr1:40000-50000',
           u'chr2:10000-20000_chr2:10000-30000',
           u'chr2:30000-50000_chr2:40000-50000',
           u'chr11:10000-20000_chr11:10000-30000',
           u'chr11:30000-50000_chr11:40000-50000',
           u'chrX:10000-20000_chrX:10000-30000',
           u'chrX:30000-50000_chrX:40000-50000'],
          dtype='object', name=u'loop_id')
    """
    # these are the six BED-like columns
    sort_order = COLUMN_ORDER[1:7]

    # we can't sort directly on *_chrom
    # we will add these two surrogate columns to sort on instead
    sort_order[0] = 'us_chrom_idx'
    sort_order[3] = 'ds_chrom_idx'

    # create a mapping from chromosome names to their index in the sort order
    idx_to_chrom = sorted(list(set(list(cluster_table['us_chrom'].unique()) +
                                   list(cluster_table['ds_chrom'].unique()))),
                          key=natural_sort_key)
    chrom_to_idx = {idx_to_chrom[i]: i for i in range(len(idx_to_chrom))}

    # use this mapping to add our surrogate columns
    cluster_table['us_chrom_idx'] = cluster_table['us_chrom']\
        .replace(chrom_to_idx)
    cluster_table['ds_chrom_idx'] = cluster_table['ds_chrom']\
        .replace(chrom_to_idx)

    # sort, then drop the surrogate columns before returning
    return cluster_table.sort_values(sort_order)\
        .drop(columns=['us_chrom_idx', 'ds_chrom_idx'])


def load_cluster_table(table_filename):
    r"""
    Loads a cluster table from a TSV file on disk to a DataFrame.

    This function will ensure that the "cluster" column of the DataFrame is
    converted from a string representation to a list of list of int to simplify
    downstream processing.

    See the example below for details on how this function assumes the cluster
    table was saved.

    Parameters
    ----------
    table_filename : str
        String reference to the location of the TSV file.

    Returns
    -------
    pd.DataFrame
        The loaded cluster table.

    Examples
    --------
    >>> from io import BytesIO as StringIO
    >>> from hic3defdr.util.cluster_table import clusters_to_table, \
    ...     load_cluster_table
    >>> clusters = [[(1, 2), (1, 1)], [(4, 4),  (3, 4)]]
    >>> df = clusters_to_table(clusters, 'chrX', 10000)
    >>> s = StringIO()  # simulates a file on disk
    >>> df.to_csv(s, sep='\t')
    >>> position = s.seek(0)
    >>> loaded_df = load_cluster_table(s)
    >>> df.equals(loaded_df)
    True
    >>> loaded_df['cluster'][0]
    [[1, 2], [1, 1]]
    """
    df = pd.read_csv(table_filename, sep='\t', index_col=0)
    df['cluster'] = df['cluster'].apply(cluster_from_string)
    return df
