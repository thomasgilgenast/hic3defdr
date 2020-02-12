# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project attempts to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2020-02-12

### Added
 - This changelog!
 - The pipeline now generates a final table of loop calls as requested by [#23](https://bitbucket.org/creminslab/hic3defdr/issues/23/pipeline-should-write-a-table-of-final).
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
   see [#28](https://bitbucket.org/creminslab/hic3defdr/issues/28/attributeerror-module-object-has-no)

### Fixed
 - [#26](https://bitbucket.org/creminslab/hic3defdr/issues/26/hic3defdr-util-lowesspy-179-runtimewarning)
 - [#27](https://bitbucket.org/creminslab/hic3defdr/issues/27/valueerror-could-not-convert-string-to)

## 0.0.9 - 2019-11-26

First official release, corresponds to what was used in the second submission of
the related manuscript.

## Diffs
- [0.1.0](https://bitbucket.org/creminslab/hic3defdr/branches/compare/0.1.0..0.0.9#diff)
- [0.0.9](https://bitbucket.org/creminslab/hic3defdr/src/0.0.9)
