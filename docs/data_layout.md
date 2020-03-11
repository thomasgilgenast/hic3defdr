hic3defdr data layout
=====================

In brief, the hic3defdr data layout is like a COO-format sparse matrix where the
data vector is actually a rectangle, storing parallel data vectors for multiple
replicates in the same data structure. This allows hic3defdr to combine the
advantages of sparse matrix formats (like COO) together with the applications of
analyzing data across replicates (like differential loop calling).

Like COO, the hic3defdr data layout keeps track of a `row` and `col` vector for
each chromosome. These vectors are stored on disk as `<outdir>/row_<chrom>.npy`
and `<outdir>/col_<chrom>.npy`, respectively.

In contrast to COO, where the `data` vector (parallel to `row` and `col`) is
just a vector, in the hic3defdr data layout the `data` can be a rectangular
matrix whose rows correspond to pixels (same length as `row` and `col`) and
whose columns correspond to replicates or conditions. Each stage of the
hic3defdr pipeline writes its output (in the form of this rectangle) to disk as
`<outdir>/<stage>_<chrom>.npy`. The hic3defdr data layout is designed so that
multiple "stages" of data processing can re-use the same `row` and `col`
vectors, making it easy to trace pixel values across stages as well as across
replicates.

One important complication is that certain steps of data processing may filter
out pixels from the pipeline. This means that the number of pixels (number of
rows in the rectangular `data` matrix) may be smaller for the output of later
pipeline steps. Since these matrices have fewer rows, they don't align with the
`row` and `col` vectors, which are always the same length. To address this
problem, boolean index vectors stored on disk as e.g.
`<outdir>/disp_idx_<chrom>.npy` are aligned with `row` and `col` and are True at
all pixels that are kept during filtering. This means that `row[disp_idx]` is
aligned with rectangular matrices after the `disp_idx` filtering step, such as
`<outdir>/disp_<chrom>.npy`. Finally, these indices can be chained, so that
`row[disp_idx][loop_idx]` is aligned with rectangular matrices after the
`loop_idx` filtering step, such as `<outdir>/qvalues_<chrom>.npy`.

A complete table of all the outputs, their expected shapes, and what boolean
indices are needed to align them to `row` and `col` is provided in the README
section "Intermediates and final output files".
