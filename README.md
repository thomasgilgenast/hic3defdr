fast3defdr
==========

a genome-scale differential loop finder

Installation
------------

We require Python 2.7 and the dependencies listed in `setup.py`.

A typical quick install process should be:

    $ virtualenv venv
    $ source venv/bin/activate
    (venv)$ pip install numpy
    (venv)$ pip install git+https://<username>@bitbucket.org/creminslab/fast3defdr.git

A typical dev-mode install process should be:

    $ git clone https://<username>@bitbucket.org/creminslab/fast3defdr.git
    $ cd fast3defdr
    $ virtualenv venv
    $ source venv/bin/activate
    (venv)$ pip install numpy
    (venv)$ pip install -e .

If installation succeeded then `fast3defdr.Fast3DeFDR` should be importable from
an interactive shell started in some other directory:

    (venv)$ cd <some other directory>
    (venv)$ python
    >>> from fast3defdr import Fast3DeFDR

Basic walkthrough
-----------------

To analyze the ES_1, ES_3, NPC_2, and NPC_4 reps of the Bonev dataset with
default parameters, we would first describe the dataset in terms of replicate
names, chromosome names, and a design matrix:

    >>> import pandas as pd
    >>>
    >>> repnames = ['ES_1', 'ES_3', 'NPC_2', 'NPC_4']
    >>> chroms = ['chr%i' % i for i in range(1, 20)] + ['chrX']
    >>> design = pd.DataFrame({'ES': [1, 1, 0, 0], 'NPC': [0, 0, 1, 1]},
    ...                       dtype=bool, index=repnames)

We would next describe the location of input data files (raw contact matrices in
`scipy.sparse` NPZ format, bias vectors in plain-text `np.savetxt()` format,
and loop cluster files in sparse JSON format) and use those to construct a
`Fast3DeFDR` object:

    >>> from fast3defdr import Fast3DeFDR
    >>>
    >>> base_path = '...'
    >>> f = Fast3DeFDR(
    ...     raw_npz_patterns=[base_path + '<rep>/<chrom>_raw.npz'.replace('<rep>', repname) for repname in repnames],
    ...     bias_patterns=[base_path + '<rep>/<chrom>_kr.bias'.replace('<rep>', repname) for repname in repnames],
    ...     chroms=chroms,
    ...     design=design,
    ...     outdir='output',
    ...     loop_patterns=[base_path + 'clusters/%s_<chrom>_clusters.json' % c for c in ['ES', 'NPC']]
    ... )

This object saves itself to disk, so it can be re-loaded at any time:

    >>> f = Fast3DeFDR.load('output')

To run the analysis for all chromosomes through q-values, run:

    >>> f.run_to_qvalues()

To threshold, cluster, and classify the significantly differential loops, run:

    >>> f.classify()

Step-by-step walkthrough
------------------------

We prepare the input data and compute the size factors with:

    >>> f.prepare_data()

We estimate dispersions with:

    >>> f.estimate_disp()

We perform the likelihood ratio test to obtain p-values with:

    >>> f.lrt()

We apply BH-FDR correction to the p-values across all chromosomes to obtain 
q-values:

    >>> f.bh()

We threshold, cluster, and classify the significantly differential loops:

    >>> f.classify(fdr=0.05, cluster_size=3)

We can also sweep across FDR and/or cluster size thresholds:

    >>> f.classify(fdr=[0.01, 0.05], cluster_size=[3, 4])

`f.classify()` calls `f.threshold()` automatically for FDR and cluster size 
thresholds that have not been run yet. `f.threshold()` is the step that performs 
thresholding and clustering but not classification.

The complete analysis should take about 10 minutes on a laptop and fit
comfortably in memory.

Intermediates and final output files
------------------------------------

All intermediates used in the computation will be saved to the disk inside the
`outdir` folder as `<intermediate>_<chrom>.npy` or `<intermediate>_<chrom>.json`

| Step              | Intermediate    | Shape                       | Description                                 |
|-------------------|-----------------|-----------------------------|---------------------------------------------|
| `prepare_data()`  | `row`           | `(n_pixels,)`               | Top-level row index                         |
| `prepare_data()`  | `col`           | `(n_pixels,)`               | Top-level column index                      |
| `prepare_data()`  | `bias`          | `(n_bins, n_reps)`          | Bias vectors                                |
| `prepare_data()`  | `raw`           | `(n_pixels, n_reps)`        | Raw count values                            |
| `prepare_data()`  | `size_factors`  | `(n_reps,)`                 | Size factors                                |
| `prepare_data()`  | `scaled`        | `(n_pixels, n_reps)`        | Normalized count values                     |
| `estimate_disp()` | `disp_idx`      | `(n_pixels,)`               | Marks pixels for which dispersion is fitted |
| `estimate_disp()` | `cov_per_bin`   | `(n_bins, n_conds)`         | Average mean count or distance in each bin  |
| `estimate_disp()` | `disp_per_bin`  | `(n_bins, n_conds)`         | Pooled dispersion estimates in each bin     |
| `estimate_disp()` | `disp`          | `(disp_idx.sum(), n_conds)` | Smoothed dispersion estimates               |
| `lrt()`           | `mu_hat_null`   | `(disp_idx.sum(),)`         | Null model mean parameters                  |
| `lrt()`           | `mu_hat_alt`    | `(disp_idx.sum(), n_conds)` | Alternative model mean parameters           |
| `lrt()`           | `llr`           | `(disp_idx.sum(),)`         | Log-likelihood ratio                        |
| `lrt()`           | `pvalues`       | `(disp_idx.sum(),)`         | LRT-based p-value                           |
| `lrt()`           | `loop_idx`      | `(disp_idx.sum(),)`         | Marks pixels which lie in loops             |
| `bh()`            | `qvalues`       | `(loop_idx.sum(),)`         | BH-corrected q-values                       |
| `threshold()`     | `sig_<f>_<s>`   | JSON                        | Significantly differential clusters         |
| `threshold()`     | `insig_<f>_<s>` | JSON                        | Constitutive clusters                       |
| `classify()`      | `<c>_<f>_<s>`   | JSON                        | Classified differential clusters            |

The table uses these abbreviations to refer to variable parts of certain 
intermediate names:

 - `<f>`: FDR threshold
 - `<s>`: cluster size threshold
 - `<c>`: condition/class label

TODO: add a tsv-style output file

Visualizations
--------------

The `Fast3DeFDR` object can be used to draw visualizations of the analysis.

### Distance dependence curves before and after scaling

    >>> f.plot_dd_curves('chr1', outfile='dd.png')

![](images/dd.png)

### Dispersion fitting

    >>> f.plot_dispersion_fit('chr1', 'ES', outfile='disp.png')

![](images/disp.png)

    >>> f.plot_dispersion_fit('chr1', 'ES', yaxis='var', outfile='var.png')

![](images/var.png)

### P-value distribution

    >>> f.plot_pvalue_distribution(outfile='pvalue_dist.png')

![](images/pvalue_dist.png)

### Q-value distribution

    >>> f.plot_qvalue_distribution(outfile='qvalue_dist.png')

![](images/qvalue_dist.png)

### Pixel detail grid

    >>> f.plot_grid('chr1', 1303, 1312, 50, outfile='grid.png')

![](images/grid.png)

The upper right heatmaps show the balanced and scaled values in each replicate,
with each condition on its own row.

The upper left heatmaps show the alternative model mean parameter estimates for
each condition. Significantly differential clusters are purple while
constitutive ones are gray.

The lower left heatmap shows the q-values. Significantly differential clusters
are orange while constitutive ones are gray.

The stripplots in the lower left show details information about the specific
pixel in the center of the heatmaps (in this example `(1303, 1312)`). The dots
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
    from fast3defdr import Fast3DeFDR
    
    f = Fast3DeFDR.load('output')
    _, _, outline_clusters = f.plot_grid('chr1', 1303, 1312, 50)
    _ = interact(outline_clusters, fdr=[0.01, 0.05, 0.1, 0.2],
                 cluster_size=[3, 4])
