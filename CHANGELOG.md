# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project attempts to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.1 - 2020-05-03

### Added
 - A license (MIT).
 - Utilty functions for APA plots in `hic3defdr.util.apa` with a corresponding
   demo in `docs/apa.rst`.
 - Utility function for importing loop calls from HiCCUPS in
   `hic3defdr.util.clusters.hiccups_to_clusters()`.

### Updates/maintenance
 - Demos now run as tests within the readme tox environments.
 - Release pipeline no longer creates any artifacts, see comments on [this commit](https://bitbucket.org/creminslab/hic3defdr/commits/df8ff2a).

## 0.2.0 - 2020-03-13

This version adds Python 3 support!

### Added
 - A new utility module to load demo data (`hic3defdr.util.demo_data`), see
   [#39](https://bitbucket.org/creminslab/hic3defdr/issues/39).
 - New utility modules to handle balancing of simulated datasets
   (`hic3defdr.util.balancing`, `hic3defdr.util.banded_matrix`, and
   `hic3defdr.util.filtering`), see
   [#40](https://bitbucket.org/creminslab/hic3defdr/issues/40).

### Changed
 - numpy arrays containing strings now use a "U"-based dtype.

### Updates/maintenance
 - Bumped minimum lib5c dependency to 0.6.0.
 - Version information is now provided by [setuptools-scm](https://pypi.org/project/setuptools-scm/)
   instead of versioneer.
 - Added Sphinx documentation and readthedocs configuration to build it.
 - Linting, testing, running the README, and building docs are now all handled
   using [tox](https://tox.readthedocs.io/).
 - Reworked Docker image build, see [creminslab/lib5c#57](https://bitbucket.org/creminslab/lib5c/issues/67).

## 0.1.1 - 2020-02-27

Streamlined data loading, see [#19](https://bitbucket.org/creminslab/hic3defdr/issues/19).

### Added
 - A new convenience function `HiC3DeFDR.get_matrix()` which selects a dense
   matrix slice for any stage of the data in one line.
 - A new convenience kwarg `coo` on `HiC3DeFDR.load_data()` which (when True)
   causes the function to return a COO-format `row, col, data` tuple that's
   guaranteed to be aligned for any stage of the data.
 - New convenience kwargs `rep` and `cond` on `HiC3DeFDR.load_data()` which
   allow selecting the right column of rectangular data stages by name
   automatically (i.e., without having to manually compute `rep_idx` ).
 - The `stage` kwarg of `HiC3DeFDR.plot_heatmap()` can now be any stage,
   including 'qvalues'. This allows easy plotting of heatmaps showing the
   significance of each pixel. An example showing how to do this has been added
   to the README.

### Changed
 - The signature of `HiC3DeFDR.plot_heatmap()` has been reworked so that `rep`
   is now a kwarg instead of the first positional arg.
 - The signature of `hic3defdr.plotting.heatmap.plot_heatmap()` has been
   reworked so that it accepts a dense matrix `matrix` as the first positional
   arg instead of `row, col, data, row_slice, col_slice`. Clients are expected
   to use `hic3defdr.util.matrix.select_matrix()` to get the dense matrix before
   calling `plot_heatmap()`.
 - `HiC3DeFDR.load_data('loop_idx', ...)` now returns a vector of True if
   ``loop_patterns`` was not passed to the HiC3DeFDR constructor.

## 0.1.0 - 2020-02-12

Adds a first draft of TSV output tables.

### Added
 - This changelog!
 - The pipeline now generates a final table of loop calls as requested by [#23](https://bitbucket.org/creminslab/hic3defdr/issues/23).
    - Added a new pipeline step `HiC3DeFDR.collect()` - we now recommend running
      this instead of `HiC3DeFDR.classify()`.
    - Added a new utility module `hic3defdr.util.cluster_table` to support
      creating tabular files summarizing cluster information
    - Added a few new functions in `hic3defdr.util.clusters` as well as some
      better test coverage in this module.
    - The README has been updated to cover this new step and output format.

### Changed
 - Renamed `hic3defdr.util.logging` to `hic3defdr.util.printing` to avoid a rare
   name clash bug related to [this issue](https://github.com/pandas-dev/pandas/issues/10167),
   see [#28](https://bitbucket.org/creminslab/hic3defdr/issues/28)

### Fixed
 - [#26](https://bitbucket.org/creminslab/hic3defdr/issues/26)
 - [#27](https://bitbucket.org/creminslab/hic3defdr/issues/27)

## 0.0.9 - 2019-11-26

First official release, corresponds to what was used in the second submission of
the related manuscript.

## Diffs
- [0.2.1](https://bitbucket.org/creminslab/hic3defdr/branches/compare/0.2.1..0.2.0#diff)
- [0.2.0](https://bitbucket.org/creminslab/hic3defdr/branches/compare/0.2.0..0.1.1#diff)
- [0.1.1](https://bitbucket.org/creminslab/hic3defdr/branches/compare/0.1.1..0.1.0#diff)
- [0.1.0](https://bitbucket.org/creminslab/hic3defdr/branches/compare/0.1.0..0.0.9#diff)
- [0.0.9](https://bitbucket.org/creminslab/hic3defdr/src/0.0.9)
