hic3defdr
=========

a genome-scale differential loop finder

Installation
------------

We require Python 2.7 and the dependencies listed in `setup.py`.

A typical quick install process should be:

    $ virtualenv venv
    $ source venv/bin/activate
    (venv)$ pip install numpy
    (venv)$ pip install --extra-index-url https://pypi.gilgi.org hic3defdr

A typical dev-mode install process should be:

    $ git clone https://<username>@bitbucket.org/creminslab/hic3defdr.git
    $ cd hic3defdr
    $ virtualenv venv
    $ source venv/bin/activate
    (venv)$ pip install numpy
    (venv)$ pip install -e .

If installation succeeded then `hic3defdr.HiC3DeFDR` should be importable from
an interactive shell started in some other directory:

    (venv)$ cd <some other directory>
    (venv)$ python
    >>> from hic3defdr import HiC3DeFDR

### Optional dependencies

Evaluating simulations requires scikit-learn:

    (venv)$ pip install scikit-learn

To display progress bars during selected steps of the analysis, install [tqdm](https://github.com/tqdm/tqdm):

    (venv)$ pip install tqdm

To execute tests, install the following:

    (venv)$ pip install nose nose-exclude flake8

Basic walkthrough
-----------------

Before we start, we'll seed numpy's random number generator for reproducibility:

    >>> import numpy as np
    >>> np.random.seed(42)

To analyze the ES_1, ES_3, NPC_2, and NPC_4 reps of the dataset dataset from
[Bonev et al. 2017](https://www.ncbi.nlm.nih.gov/pubmed/29053968) with default
parameters, we would first describe the dataset in terms of replicate names,
chromosome names, and a design matrix. We will just analyze chromosomes 18 and
19 here for illustrative purposes.

    >>> import pandas as pd
    >>>
    >>> repnames = ['ES_1', 'ES_3', 'NPC_2', 'NPC_4']
    >>> #chroms = ['chr%i' % i for i in range(1, 20)] + ['chrX']
    >>> chroms = ['chr18', 'chr19']
    >>> design = pd.DataFrame({'ES': [1, 1, 0, 0], 'NPC': [0, 0, 1, 1]},
    ...                       dtype=bool, index=repnames)

If you're following along, you can download the data like this:

    $ mkdir -p ~/data/bonev
    $ wget -qO- -O tmp.zip https://www.dropbox.com/sh/hvoyhjc00m24o6m/AAAci5qaxsn7o9W-gToAeBiza?dl=1 && unzip tmp.zip -x / -d ~/data/bonev && rm tmp.zip

The required input files consist of:

 - upper triangular, raw contact matrices in `scipy.sparse` NPZ format,
 - bias vectors in plain-text `np.savetxt()` format, and
 - loop cluster files in sparse JSON format (see below for more details),
   specifying the locations of loops present in each condition

TODO: explain how to import data from common Hi-C analysis tools into this
format

We would next describe the location of the input data files and use those to
construct a `HiC3DeFDR` object:

    >>> import os.path
    >>> from hic3defdr import HiC3DeFDR
    >>>
    >>> base_path = os.path.expanduser('~/data/bonev/')
    >>> h = HiC3DeFDR(
    ...     raw_npz_patterns=[base_path + '<rep>/<chrom>_raw.npz'.replace('<rep>', repname) for repname in repnames],
    ...     bias_patterns=[base_path + '<rep>/<chrom>_kr.bias'.replace('<rep>', repname) for repname in repnames],
    ...     chroms=chroms,
    ...     design=design,
    ...     outdir='output',
    ...     loop_patterns={c: base_path + 'clusters/%s_<chrom>_clusters.json' % c for c in ['ES', 'NPC']}
    ... )
    creating directory output

This object saves itself to disk, so it can be re-loaded at any time:

    >>> h = HiC3DeFDR.load('output')

To run the analysis for all chromosomes through q-values, run:

    >>> h.run_to_qvalues()

To threshold, cluster, and classify the significantly differential loops, run:

    >>> h.classify()

Step-by-step walkthrough
------------------------

Calling `h.run_to_qvalues()` runs the four main steps of the analysis in
sequence. These four steps are described in further detail below. Any kwargs
passed to `h.run_to_qvalues()` will be passed along to the appropriate step; see
`help(HiC3DeFDR.run_to_qvalues())` for details.

### Step 1: Preparing input data

The function call `h.prepare_data()` prepares the input raw contact matrices and
bias vectors specified by `h.raw_npz_patterns` and `h.bias_patterns` for all
chromosomes specified in `h.chroms`, performs library size normalization, and
determines what points should be considered for dispersion estimation. This
creates intermediate files in the output directory `h.outdir` that represent the
raw and scaled values, as well as the estimated size factors, and a boolean
vector `disp_idx` indicating which points will be used for dispersion
estimation. If `loop_patterns` was passed to the constructor, an additional
boolean vector `loop_idx` is created to mark which points that pass `disp_idx`
lie within looping interactions specified by `h.loop_patterns`. The raw and
scaled data are stored in a rectangular matrix format where each row is a pixel
of the contact matrix and each column is a replicate. If the size factors are
estimated as a function of distance, the estimated size factors are also stored
in this format. Two separate vectors called `row` and `col` are used to store
the row and column index of the pixel represented by each row of the rectangular
matrices. Together, the `row` and `col` vectors plus any of the rectangular
matrices represent a generalization of a COO format sparse matrix to multiple
replicates (in the standard COO format the `row` and `col` vectors are
complemented by a single `value` vector).

The size factors can be estimated with a variety of methods defined in the
`hic3defdr.scaling` module. The method to use is specified by the `norm` kwarg
passed to `h.prepare_data()`. Some of these methods estimate size factors as a
function of interaction distance, instead of simply estimating one size factor
for each replicate as is common in RNA-seq differential expression analysis.
When these methods are used, the number of bins to use when binning distances
for distance-based estimation of the size factors can be specified with the
`n_bins` kwarg. The defaults for this function use the conditional median of
ratios approach (`hic3defdr.scaling.conditional_mor`) and an
automatically-selected number of bins.

### Step 2: Estimating dispersions

The function call `h.estimate_disp()` estimates the dispersion parameters at
each distance scale in the data and fits a lowess curve through the graph of
distance versus dispersion to obtain final smoothed dispersion estimates for
each pixel.

The `estimator` kwarg on `h.estimate_disp()` specifies which dispersion
estimation method to use, out of a selection of options defined in the
`hic3defdr.dispersion` module. The default is to use quantile-adjusted
conditional maximum likelihood (qCML).

### Step 3: Likelihood ratio test

The function call `h.lrt()` performs a likelihood ratio test (LRT) for each
pixel. This LRT compares a null model of no differential expression (fitting one
true mean parameter shared by all replicates irrespective of condition) to an
alternative model in which the two conditions have different true mean
parameters.

### Step 4: False discovery rate (FDR) control

The function call `h.bh()` performs Benjamini-Hochberg FDR correction on the
p-values called via the LRT in the previous step, considering only a subset of
pixels that are involved in looping interactions (as specified by
`h.loop_patterns`; if `loop_patterns` was not passed to the constructor then all
pixels are included in the BH-FDR correction). This results in final q-values
for each loop pixel.

### Thresholding, clustering, and classification

We threshold, cluster, and classify the significantly differential loops:

    >>> h.classify(fdr=0.05, cluster_size=3)

We can also sweep across FDR and/or cluster size thresholds:

    >>> h.classify(fdr=[0.01, 0.05], cluster_size=[3, 4])

`h.classify()` calls `h.threshold()` automatically for FDR and cluster size
thresholds that have not been run yet. `h.threshold()` is the step that performs
thresholding and clustering but not classification.

This example walkthrough should take less than 5 minutes for the two chromosomes
included in the demo data. Run time for a whole-genome analysis will depend on
parallelization as discussed in the next section.

Parallelization
---------------

The functions `run_to_qvalues()`, `prepare_data()`, `lrt()`, `threshold()`,
`classify()`, and `simulate()` operate in a per-chromosome manner. By default,
each chromosome in the dataset will be processed in series. If multiple cores
and sufficient memory are available, you can use the `n_threads` kwarg on these
functions to use multiple subprocesses to process multiple chromosomes in
parallel. Pass either a desired number of subprocesses to use, or pass
`n_threads=-1` to use all available cores. The output logged to stderr will be
truncated to reduce clutter from multiple subprocesses printing to stderr at the
same time. This truncation is controlled by the `verbose` kwarg which is
available on some of these parallelized functions.

Intermediates and final output files
------------------------------------

All intermediates used in the computation will be saved to the disk inside the
`outdir` folder as `<intermediate>_<chrom>.npy` (most intermediates),
`<intermediate>_<chrom>.json` (thresholded or classified clusters), or
`<intermediate>_<chrom>.pickle` (estimated dispersion functions).

| Step              | Intermediate    | Shape                               | Description                                 |
|-------------------|-----------------|-------------------------------------|---------------------------------------------|
| `prepare_data()`  | `row`           | `(n_pixels,)`                       | Top-level row index                         |
| `prepare_data()`  | `col`           | `(n_pixels,)`                       | Top-level column index                      |
| `prepare_data()`  | `bias`          | `(n_bins, n_reps)`                  | Bias vectors                                |
| `prepare_data()`  | `raw`           | `(n_pixels, n_reps)`                | Raw count values                            |
| `prepare_data()`  | `size_factors`  | `(n_reps,)` or `(n_pixels, n_reps)` | Size factors                                |
| `prepare_data()`  | `scaled`        | `(n_pixels, n_reps)`                | Normalized count values                     |
| `prepare_data()`  | `disp_idx`      | `(n_pixels,)`                       | Marks pixels for which dispersion is fitted |
| `prepare_data()`  | `loop_idx`      | `(disp_idx.sum(),)`                 | Marks pixels which lie in loops             |
| `estimate_disp()` | `cov_per_bin`   | `(n_bins, n_conds)`                 | Average mean count or distance in each bin  |
| `estimate_disp()` | `disp_per_bin`  | `(n_bins, n_conds)`                 | Pooled dispersion estimates in each bin     |
| `estimate_disp()` | `disp_fn_<c>`   | pickled function                    | Fitted dispersion function                  |
| `estimate_disp()` | `disp`          | `(disp_idx.sum(), n_conds)`         | Smoothed dispersion estimates               |
| `lrt()`           | `mu_hat_null`   | `(disp_idx.sum(),)`                 | Null model mean parameters                  |
| `lrt()`           | `mu_hat_alt`    | `(disp_idx.sum(), n_conds)`         | Alternative model mean parameters           |
| `lrt()`           | `llr`           | `(disp_idx.sum(),)`                 | Log-likelihood ratio                        |
| `lrt()`           | `pvalues`       | `(disp_idx.sum(),)`                 | LRT-based p-value                           |
| `bh()`            | `qvalues`       | `(loop_idx.sum(),)`                 | BH-corrected q-values                       |
| `threshold()`     | `sig_<f>_<s>`   | JSON                                | Significantly differential clusters         |
| `threshold()`     | `insig_<f>_<s>` | JSON                                | Constitutive clusters                       |
| `classify()`      | `<c>_<f>_<s>`   | JSON                                | Classified differential clusters            |

The table uses these abbreviations to refer to variable parts of certain
intermediate names:

 - `<f>`: FDR threshold
 - `<s>`: cluster size threshold
 - `<c>`: condition/class label

TODO: add a tsv-style output file

Sparse JSON cluster format
--------------------------

This is the format used both for supplying pre-identified, per-condition loop
clusters as input to HiC3DeFDR as well as the format in which differential and
constitutive interaction clusters are reported as output.

The format describes the clusters on each chromosome in a separate JSON file.
This JSON file contains a single JSON object, which is a list of list of list of
integers. The inner lists are all of length 2 and represent specific pixels of
the heatmap for that chromosome in terms of there row and column coordinates in
zero-indexed bin units. The outer middle lists can be of any length and
represent groups of pixels that belong to the same cluster. The length of the
outer list corresponds to the number of clusters on that chromosome.

These clusters can be loaded into and written from the corresponding plain
Python objects using `hic3defdr.clusters.load_clusters()` and
`hic3defdr.clusters.save_clusters()`, respectively. The plain Python objects can
be converted to and from scipy sparse matrix objects using
`hic3defdr.clusters.clusters_to_coo()` and
`hic3defdr.clusters.convert_cluster_array_to_sparse()`, respectively.

Visualizations
--------------

The `HiC3DeFDR` object can be used to draw visualizations of the analysis.

The visualization functions are wrapped with the [`@plotter` decorator](https://lib5c.readthedocs.io/en/latest/plotting/)
and therefore all support the convenience kwargs provided by that decorator
(such as `outfile`).

### Distance dependence curves before and after scaling

    >>> _ = h.plot_dd_curves('chr18', outfile='dd.png')

![](images/dd.png)

### Dispersion fitting

    >>> _ = h.plot_dispersion_fit('ES', outfile='ddr.png')

![](images/ddr.png)

You can also plot the y-axis in units of variance by plotting `yaxis='var'`:

    >>> _ = h.plot_dispersion_fit('ES', yaxis='var', outfile='var.png')

![](images/dvr.png)

It's possible to use the the one-dimensional distance dependence curve to
convert distances to means and plot these figures with mean on the x-axis. To do
this, pass `xaxis='mean'`.

    >>> _ = h.plot_dispersion_fit('ES', xaxis='mean', yaxis='var', logx=True, logy=True, outfile='mvr.png')

![](images/mvr.png)

At low mean and high distance, the distance dependence curve flattens out and
the data become more noisy, making this conversion unreliable.

It's also possible to show the dispersion fitted at just one distance scale,
overlaying the sample mean and sample variance across replicates for each pixel
as a blue hexbin plot:

    >>> _ = h.plot_dispersion_fit('ES', distance=25, hexbin=True, xaxis='mean', yaxis='var', logx=True, logy=True, outfile='mvr_25.png')

![](images/mvr_25.png)

If dispersion was fitted against distance rather than mean, pass `xaxis='dist'`
to plot dispersion/variance versus distance.

### P-value distribution

    >>> _ = h.plot_pvalue_distribution(outfile='pvalue_dist.png')

![](images/pvalue_dist.png)

By default, this plots the p-value distribution over all pixels for which
dispersion was estimated. To plot the p-value distribution only over points in
loops, pass `idx='loop'`.

### Q-value distribution

    >>> _ = h.plot_qvalue_distribution(outfile='qvalue_dist.png')

![](images/qvalue_dist.png)

### MA plot

    >>> _ = h.plot_ma(legend=True, outfile='ma.png')

![](images/ma.png)

To include non-loop pixels, pass `include_non_loops=True`.

### Pixel detail grid

    >>> _ = h.plot_grid('chr18', 2218, 2236, 50, outfile='grid.png')

![](images/grid.png)

The upper right heatmaps show the balanced and scaled values in each replicate,
with each condition on its own row.

The upper left heatmaps show the alternative model mean parameter estimates for
each condition. Significantly differential clusters are purple while
constitutive ones are gray.

The lower left heatmap shows the q-values. Significantly differential clusters
are orange while constitutive ones are gray.

The stripplots in the lower left show details information about the specific
pixel in the center of the heatmaps (in this example `(2218, 2236)`). The dots
show the values at that pixel for each replicate in normalized and raw space,
repsectively. The solid and dashed lines represent the mean parameters under the
alt and null models, repsectively.

Green points in the heatmaps represent points that have been filtered out. For
the per-replicate heatmaps in the upper right of the grid, the only filters
applied are the zero filter, bias filter, and distance filter. For the alt model
mean heatmaps in the upper left, this additionally includes the dispersion
filter. For the q-value heatmap in the lower left, it additionally includes the
loop filter if loop locations were supplied.

### Interactive thresholding

In a Jupyter notebook environment with `ipywidgets` installed, you can play with
thresholds on a live-updating plot by running:

    %matplotlib notebook

    from ipywidgets import interact
    from hic3defdr import HiC3DeFDR
    
    h = HiC3DeFDR.load('output')
    _, _, outline_clusters = h.plot_grid('chr18', 2218, 2236, 50)
    _ = interact(outline_clusters, fdr=[0.01, 0.05, 0.1, 0.2],
                 cluster_size=[3, 4])

Simulation
----------

After the `estimate_disp()` step has been run, a HiC3DeFDR object with exactly
two conditions and an equal number of replicates per condition can be used to
generate simulations of differential looping.

### Generating simulations

To create an ES-based simulation over all chromosomes listed in `h.chroms`, we
run

    >>> from hic3defdr import HiC3DeFDR
    >>>
    >>> h = HiC3DeFDR.load('output')
    >>> h.simulate('ES')
    creating directory sim

If we passed `trend='dist'` to `h.estimate_disp()`, we need to pass it to
`h.simulate()` as well to ensure that the simulation function knows to treat the
previously-fitted dispersion function as a function of distance.

This takes the mean of the real scaled data across the ES replicates and
perturbs the loops specified in `h.loop_patterns['ES']` up or down at random to
generate two new conditions called "A" and "B". The scaled mean matrices for
these conditions are then biased and scaled by the bias vectors and size factors
taken from the real experimental replicates, and the ES dispersion function
fitted to the real ES data is applied to the biased and scaled means to obtain
dispersion values. These means and dispersions are used to draw an NB random
variable for each pixel of each simulated replicate. The number of replicates in
each of the simulated conditions "A" and "B" will match the design of the real
analysis.

The simulated raw contact matrices will be written to disk in CSR format as
`<cond><rep>_<chrom>_raw.npz` where `<cond>` is "A" or "B" and `<rep>` is the
rep number within the condition. The design matrix will also be written to disk
as `design.csv`.

The true labels used to perturb the loops will also be written to disk as
`labels_<chrom>.txt`. This file contains as many lines as there are clusters in
`h.loop_patterns['ES']`, with the `i`th line providing the label for the `i`th
cluster. This file can be loaded with `np.loadtxt(..., dtype='|S7')`.

### Evaluating simulations

After generating simulated data, HiC3DeFDR can be run on the simulated data.
Then, the true labels can be used to evaluate the performance of HiC3DeFDR on
the simulated data.

Evaluation of simulated data requires scikit-learn. To install this package, run

    (venv)$ pip install scikit-learn

In order to run HiC3DeFDR on the simulated data, we first need to balance the
simulated raw contact matrices to obtain bias vectors for each simulated
replicate and chromosome. We will assume are saved next to the raw contact
matrices and named `<rep>_<chrom>_kr.bias`. One example of how this can be done
is to use the [hiclite library](https://bitbucket.org/creminslab/hiclite) and
the following script:

    >>> import sys
    >>>
    >>> import numpy as np
    >>> import scipy.sparse as sparse
    >>>
    >>> from hiclite.steps.filter import filter_sparse_rows_count
    >>> from hiclite.steps.balance import kr_balance
    >>>
    >>>
    >>> infile_pattern = 'sim/<rep>_<chrom>_raw.npz'
    >>> repnames = ['A1', 'A2', 'B1', 'B2']
    >>> chroms = ['chr18', 'chr19']
    >>>
    >>> for repname in repnames:
    ...     for chrom in chroms:
    ...         sys.stderr.write('balancing rep %s chrom %s\n' % (repname, chrom))
    ...         infile = infile_pattern.replace('<rep>', repname)\
    ...             .replace('<chrom>', chrom)
    ...         outfile = infile.replace('_raw.npz', '_kr.bias')
    ...         _, bias, _ = kr_balance(
    ...             filter_sparse_rows_count(sparse.load_npz(infile)), fl=0)
    ...         np.savetxt(outfile, bias)

Next, we create a new HiC3DeFDR object to analyze the simulated data and run the
analysis through to q-values:

    >>> import os.path
    >>> from hic3defdr import HiC3DeFDR
    >>>
    >>> repnames = ['A1', 'A2', 'B1', 'B2']
    >>> chroms = ['chr18', 'chr19']
    >>> sim_path = 'sim/'
    >>> base_path = os.path.expanduser('~/data/bonev/')
    >>> h_sim = HiC3DeFDR(
    ...     raw_npz_patterns=[sim_path + '<rep>_<chrom>_raw.npz'.replace('<rep>', repname) for repname in repnames],
    ...     bias_patterns=[sim_path + '<rep>_<chrom>_kr.bias'.replace('<rep>', repname) for repname in repnames],
    ...     chroms=chroms,
    ...     design=sim_path + 'design.csv',
    ...     outdir='output-sim',
    ...     loop_patterns={'ES': base_path + 'clusters/ES_<chrom>_clusters.json'}
    ... )
    creating directory output-sim
    >>> h_sim.run_to_qvalues()

Next, we can evaluate the simulation against the clusters in
`h_sim.loop_patterns['ES']` with true labels from `sim/labels_<chrom>.txt`:

    >>> h_sim.evaluate('ES', 'sim/labels_<chrom>.txt')

This writes a file in `h_sim`'s output directory called `eval.npz`. This file
can be loaded with `np.load()` and has four keys whose values are all one
dimensional vectors:

 - `'thresh'`: the thresholds (in `1 - qvalue` space) which make up the convex
   edge of the ROC curve; all other vectors are parallel to this one
 - `'fdr'`: the observed false discovery rate at each threshold
 - `'tpr'`: the observed true positive rate at each threshold
 - `'fpr'`: the observed false positive rate at each threshold

`eval.npz` files (possibly across many runs) can be visualized as ROC curves and
FDR control curves by running:

    >>> import numpy as np
    >>> from hic3defdr import plot_roc, plot_fdr
    >>>
    >>> _ = plot_roc([np.load('output-sim/eval.npz')], ['hic3defdr'], outfile='roc.png')
    >>> _ = plot_fdr([np.load('output-sim/eval.npz')], ['hic3defdr'], outfile='fdr.png')

![](images/roc.png)
![](images/fdr.png)

Multiple `eval.npz` files can be compared in the same plot by simply adding
elements to the lists in these function calls.

The ROC plot shows FPR versus TPR, with the gray diagonal line representing the
performance of random guessing. The AUROC for each curve is shown in the legend.
If only one curve is plotted, selected thresholds (in units of FDR threshold)
are annotated with black arrows.

The FDR control plot shows the observed FDR as a function of the FDR threshold.
Points below the gray diagonal line represent points at which FDR is
successfully controlled.

Additional options
------------------

Additional options are exposed as kwargs on the functions in this library. Use
`help(<function>)` to get detailed information about the options available for
any function and what these options may be used for.
