Data filtering
==============

Points are filtered out of Hi-C datasets by HiC3DeFDR in three stages:

### 1. Initial data import

This filtering step is performed during the `prepare_data()` step.

Our motivation is to include as many points as possible. We refuse to filter out
points that have zero in one replicate (leads to underestimation of
variance/dispersion).

We

 - exclude points beyond `HiC3DeFDR.dist_thresh_max`
 - exclude points that have zero in all reps ("pixel union strategy" implemented
   in `hic3defdr.util.matrices.sparse_union()`)
 - exclude points in rows that failed balancing (decided by
   `HiC3DeFDR.bias_thresh`)

Points present in the data files `row_<chrom>.npy`, `col_<chrom>.npy`,
`raw_<chrom>.npy`, and `scaled_<chrom>.npy` reflect points that survive this
filtering step.

### 2. `disp_idx`

This filter is computed during the `prepare_data()` step, but is not used until
the `estimate_disp()` step.

Our motivation is to not try to fit dispersion to points where think dispersion
estimation will be very hard. This include points very close to the diagonal
where Hi-C gets crazy and points with very low coverage (low mean across reps).

We

 - exclude points within `HiC3DeFDR.dist_thresh_min`
 - exclude points whose mean across reps is below `HiC3DeFDR.mean_thresh`

`disp_idx_<chrom>.npy` is a boolean vector aligned to `<row/col>_<chrom>.npy`
that is True for all points that survive this filtering step.

### 3. `loop_idx`

This filter is computed during the `prepare_data()` step, but is not used until
the `bh()` step.

This filter is only computed and used if `HiC3DeFDR.loop_patterns` is not None.

Our motivation is to reduce the number of hypotheses that we test when
performing multiple testing correction via BH-FDR. Since we are only interested
in finding differential loops, we can choose to only test the hypotheses that
correspond to positions where there are loops.

We

 - exclude points that are not in a loop as defined by `HiC3DeFDR.loop_patterns`

`loop_idx_<chrom>.npy` is a boolean vector aligned to the positions where
`disp_idx_<chrom>.npy` is True that is True for all points that survive this
filtering step.
