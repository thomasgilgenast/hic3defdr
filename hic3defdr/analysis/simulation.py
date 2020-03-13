import os

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from lib5c.util.system import check_outdir
from lib5c.util.statistics import adjust_pvalues

from hic3defdr.util.printing import eprint
from hic3defdr.util.clusters import load_clusters
from hic3defdr.util.simulation import simulate
from hic3defdr.util.evaluation import make_y_true, evaluate
from hic3defdr.util.progress import tqdm_maybe as tqdm
from hic3defdr.util.parallelization import parallel_apply


class SimulatingHiC3DeFDR(object):
    """
    Mixin class containing plotting functions for HiC3DeFDR.
    """
    def simulate(self, cond, chrom=None, beta=0.5, p_diff=0.4, skip_bias=False,
                 loop_pattern=None, outdir='sim', n_threads=-1, verbose=True):
        """
        Simulates raw contact matrices based on previously fitted scaled means
        and dispersions in a specific condition.

        Can only be run after ``estimate_dispersions()`` has been run.

        Parameters
        ----------
        cond : str
            Name of the condition to base the simulation on.
        chrom : str, optional
            Name of the chromosome to simulate. Pass None to simulate all
            chromosomes in series.
        beta : float
            The effect size of the loop perturbations to use when simulating.
            Perturbed loops will be strengthened or weakened by this fraction of
            their original strength.
        p_diff : float or list of float
            Pass a single float to specify the probability that a loop will be
            perturbed across the simulated conditions. Pass four floats to
            specify the probabilities of all four specific perturbations: up in
            A, down in A, up in B, down in B. The remaining loops will be
            constitutive.
        skip_bias : bool
            Pass True to set all bias factors and size factors to 1,
            effectively simulating "unbiased" raw data.
        loop_pattern : str, optional
            File path pattern to sparse JSON formatted cluster files
            representing loop cluster locations for the simulation. Should
            contain at least one '<chrom>' which will be replaced with the
            chromosome name when loading data for specific chromosomes. Pass
            None to use ``self.loop_patterns[cond]``.
        outdir : str
            Path to a directory to store the simulated data to.
        n_threads : int
            The number of threads (technically GIL-avoiding child processes) to
            use to process multiple chromosomes in parallel. Pass -1 to use as
            many threads as there are CPUs. Pass 0 to process the chromosomes
            serially.
        verbose : bool
            Pass False to silence reporting of progress to stderr.
        """
        if chrom is None:
            if n_threads:
                parallel_apply(
                    self.simulate,
                    [{'cond': cond, 'chrom': c, 'beta': beta, 'p_diff': p_diff,
                      'skip_bias': skip_bias, 'loop_pattern': loop_pattern,
                      'outdir': outdir, 'verbose': False}
                     for c in self.chroms],
                    n_threads=n_threads
                )
            else:
                for chrom in self.chroms:
                    self.simulate(cond, chrom=chrom, beta=beta, p_diff=p_diff,
                                  loop_pattern=loop_pattern, outdir=outdir)
            return
        eprint('simulating data for chrom %s' % chrom)
        # resolve loop_pattern
        if loop_pattern is None:
            loop_pattern = self.loop_patterns[cond]

        # load everything
        bias = self.load_bias(chrom)[:, self.design[cond]]
        size_factors = self.load_data('size_factors', chrom)
        if len(size_factors.shape) == 2:
            size_factors = size_factors[:, self.design[cond]]
        else:
            size_factors = size_factors[self.design[cond]]
        row = self.load_data('row', chrom)
        col = self.load_data('col', chrom)
        scaled = self.load_data('scaled', chrom)[:, self.design[cond]]
        disp_fn = self.load_disp_fn(cond)
        clusters = load_clusters(loop_pattern.replace('<chrom>', chrom))

        # compute pixel-wise mean of normalized data
        mean = np.mean(scaled, axis=1)

        # book keeping
        check_outdir('%s/' % outdir)
        n_sim_per_cond = size_factors.shape[-1]
        repnames = sum((['%s%i' % (c, i+1) for i in range(n_sim_per_cond)]
                        for c in ['A', 'B']), [])

        # write design to disk if not present
        design_file = '%s/design.csv' % outdir
        if not os.path.isfile(design_file):
            pd.DataFrame(
                {'A': [1]*n_sim_per_cond + [0]*n_sim_per_cond,
                 'B': [0]*n_sim_per_cond + [1]*n_sim_per_cond},
                dtype=bool,
                index=repnames
            ).to_csv(design_file)

        # rewrite size_factor matrix in terms of distance
        if len(size_factors.shape) == 2:
            eprint('  converting size factors', skip=not verbose)
            dist = col - row
            n_dists = dist.max() + 1
            new_size_factors = np.zeros((n_dists, size_factors.shape[1]))
            for d in tqdm(range(n_dists)):
                idx = np.argmax(dist == d)
                new_size_factors[d, :] = size_factors[idx, :]
            size_factors = new_size_factors

        # get rid of bias
        if skip_bias:
            bias = np.ones_like(bias)
            size_factors = np.ones_like(size_factors)

        # tile bias and size_factors
        bias = np.tile(bias, 2)
        size_factors = np.tile(size_factors, 2)

        # simulate and save
        classes, sim_iter = simulate(
            row, col, mean, disp_fn, bias, size_factors, clusters, beta=beta,
            p_diff=p_diff, trend='dist', verbose=verbose)
        np.savetxt('%s/labels_%s.txt' % (outdir, chrom), classes, fmt='%s')
        for rep, csr in zip(repnames, sim_iter):
            sparse.save_npz('%s/%s_%s_raw.npz' % (outdir, rep, chrom), csr)

    def evaluate(self, cluster_pattern, label_pattern, min_dist=None,
                 max_dist=None, rerun_bh=False, outfile=None):
        """
        Evaluates the results of this analysis, comparing it to true labels.

        Parameters
        ----------
        cluster_pattern : str
            File path pattern to sparse JSON formatted cluster files
            representing loop cluster locations. Should contain at least one
            '<chrom>' which will be replaced with the chromosome name when
            loading data for specific chromosomes. Pass a condition name to use
            ``self.loop_patterns[cluster_pattern]`` instead.
        label_pattern : str
            File path pattern to true label files for each chromosome. Should
            contain at least one '<chrom>' which will be replaced with the
            chromosome name when loading data for specific chromosomes. Files
            should be loadable with ``np.loadtxt(..., dtype='U7')`` to yield a
            vector of true labels parallel to the clusters pointed to by
            ``cluster_pattern``.
        min_dist, max_dist : int, optional
            Specify minimum and maximum distances to evaluate performance
            within, respectively. Pass None to leave one or both ends unbounded.
        rerun_bh : bool
            If ``min_dist`` and/or ``max_dist`` are used to constrain the
            distances, pass True to re-run BH-FDR on the subset of p-values at
            the selected distances. Pass False to use the original dataset-wide
            q-values. Does nothing if ``min_dist`` and ``max_dist`` are both
            None.
        outfile : str, optional
            Name of a file to save the evaluation results to inside this
            object's ``outdir``. Default is 'eval.npz' if ``min_dist`` and
            ``max_dist`` are both None, otherwise it is
            'eval_<min_dist>_<max_dist>.npz'.
        """
        # resolve outfile
        if outfile is None:
            if min_dist is None and max_dist is None:
                outfile = 'eval.npz'
            else:
                outfile = 'eval_%s_%s.npz' % (min_dist, max_dist)

        # resolve case where a condition name was passed to cluster_pattern
        if cluster_pattern in self.loop_patterns.keys():
            cluster_pattern = self.loop_patterns[cluster_pattern]

        # make y_true and pvalues/qvalues (if necessary) one chrom at a time
        y_true = []
        pvalues = []
        qvalues = []
        for chrom in self.chroms:
            # load data
            disp_idx = self.load_data('disp_idx', chrom)
            loop_idx = self.load_data('loop_idx', chrom)
            row = self.load_data('row', chrom, idx=(disp_idx, loop_idx))
            col = self.load_data('col', chrom, idx=(disp_idx, loop_idx))
            clusters = load_clusters(cluster_pattern.replace('<chrom>', chrom))
            labels = np.loadtxt(label_pattern.replace('<chrom>', chrom),
                                dtype='U7')

            # construct dist_idx
            dist = col - row
            dist_idx = np.ones(len(dist), dtype=bool)
            if min_dist is not None:
                dist_idx[dist < min_dist] = False
            if max_dist is not None:
                dist_idx[dist > max_dist] = False

            # append to y_true and pvalues/qvalues (if necessary)
            y_true.append(make_y_true(
                row[dist_idx], col[dist_idx], clusters, labels))
            if min_dist is not None or max_dist is not None:
                if rerun_bh:
                    pvalues.append(self.load_data('pvalues', chrom,
                                   idx=(loop_idx, dist_idx)))
                else:
                    qvalues.append(self.load_data('qvalues', chrom,
                                   idx=dist_idx))

        # concatenate y_true and make or load qvalues
        y_true = np.concatenate(y_true)
        if pvalues:
            qvalues = adjust_pvalues(np.concatenate(pvalues))
        elif qvalues:
            qvalues = np.concatenate(qvalues)
        else:
            qvalues, _ = self.load_data('qvalues', 'all')

        # evaluate and save to disk
        fdr, fpr, tpr, thresh = evaluate(y_true, qvalues)

        # save to disk
        np.savez('%s/%s' % (self.outdir, outfile),
                 **{'fdr': fdr, 'fpr': fpr, 'tpr': tpr, 'thresh': thresh})
