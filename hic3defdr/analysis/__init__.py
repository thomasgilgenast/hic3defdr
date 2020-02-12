import pandas as pd
import dill as pickle

from lib5c.util.system import check_outdir

from hic3defdr.analysis.core import CoreHiC3DeFDR
from hic3defdr.analysis.analysis import AnalyzingHiC3DeFDR
from hic3defdr.analysis.simulation import SimulatingHiC3DeFDR
from hic3defdr.analysis.plotting import PlottingHiC3DeFDR


class HiC3DeFDR(CoreHiC3DeFDR, AnalyzingHiC3DeFDR, SimulatingHiC3DeFDR,
                PlottingHiC3DeFDR):
    """
    Main object for hic3defdr analysis.

    Attributes
    ----------
    raw_npz_patterns : list of str
        File path patterns to ``scipy.sparse`` formatted NPZ files containing
        raw contact matrices for each replicate, in order. Each file path
        pattern should contain at least one '<chrom>' which will be replaced
        with the chromosome name when loading data for specific chromosomes.
    bias_patterns : list of str
        File path patterns to ``np.savetxt()`` formatted files containing bias
        vector information for each replicate, in order. ach file path pattern
        should contain at least one '<chrom>' which will be replaced with the
        chromosome name when loading data for specific chromosomes.
    chroms : list of str
        List of chromosome names as strings. These names will be substituted in
        for '<chroms>' in the ``raw_npz_patterns`` and ``bias_patterns``.
    design : pd.DataFrame or str
        Pass a DataFrame with boolean dtype whose rows correspond to replicates
        and whose columns correspond to conditions. Replicate and condition
        names will be inferred from the row and column labels, respectively. If
        you pass a string, the DataFrame will be loaded via
        ``pd.read_csv(design, index_col=0)``.
    outdir : str
        Specify a directory to store the results of the analysis. Two different
        HiC3DeFDR analyses cannot co-exist in the same directory. The directory
        will be created if it does not exist.
    dist_thresh_min, dist_thresh_max : int
        The minimum and maximum interaction distance (in bin units) to include
        in the analysis.
    bias_thresh : float
        Bins with a bias factor below this threshold or above its reciprocal in
        any replicate will be filtered out of the analysis.
    mean_thresh : float
        Pixels with mean value below this threshold will be filtered out at the
        dispersion fitting stage.
    loop_patterns : dict of str, optional
        Keys should be condition names as strings, values should be file path
        patterns to sparse JSON formatted cluster files representing called
        loops in that condition. Each file path pattern should contain at least
        one '<chrom>' which will be replaced with the chromosome name when
        loading data for specific chromosomes.
    res : int, optional
        The bin resolution, in base pair units, of the input contact matrix
        data. Used only when printing TSV output. Pass None to skip printing TSV
        output during the ``threshold()`` and ``classify()`` steps.
    """
    def __init__(self, raw_npz_patterns, bias_patterns, chroms, design, outdir,
                 dist_thresh_min=4, dist_thresh_max=200, bias_thresh=0.1,
                 mean_thresh=1.0, loop_patterns=None, res=None):
        """
        Base constructor. See ``help(HiC3DeFDR)`` for details.
        """
        self.raw_npz_patterns = raw_npz_patterns
        self.bias_patterns = bias_patterns
        self.chroms = chroms
        if type(design) == str:
            self.design = pd.read_csv(design, index_col=0)
        else:
            self.design = design
        self.outdir = outdir
        self.dist_thresh_min = dist_thresh_min
        self.dist_thresh_max = dist_thresh_max
        self.bias_thresh = bias_thresh
        self.mean_thresh = mean_thresh
        self.loop_patterns = loop_patterns
        self.res = res
        state = self.__dict__.copy()
        del state['outdir']
        check_outdir(self.picklefile)
        with open(self.picklefile, 'wb') as handle:
            pickle.dump(state, handle, -1)
