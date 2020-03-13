"""
Module for managing version naming scheme and supplying version information.

We use setuptools_scm to get version information from git. The version naming
scheme is specified by defining functions ``version_scheme()`` and
``local_scheme()`` in this module. We have written these so that they mimic
versioneer's "pep440" style. These functions are used in this module, and they
are also imported in ``setup.py`` at install time.

This module also provides the function ``get_version()``, which gets the version
if either setuptools_scm or our package is installed, and returns "unknown"
otherwise. Getting the version from setuptools_scm is primarily useful for
Docker image building (where it doesn't make sense to make the host install our
package just to obtain the version information to pass to the image build
process) and for editable installs (where having setuptools_scm installed is the
only way to get accurate version information).

When our package is installed, its version (the result of ``get_version()``) can
be accessed as ``<our_package>.__version__``. This is the version printed when
``<our_package> -v`` is run on the command line.

This module can be run as a script to print version information to stdout.
"""


def version_scheme(version):
    return str(version.tag)


def local_scheme(version):
    if version.distance is None:
        return ''
    return '+%s.%s%s' % (version.distance, version.node,
                         '.dirty' if version.dirty else '')


def get_version():
    try:
        # our first choice would be to get the version from setuptools_scm if it
        # is installed (only way that works with editable installs)
        from setuptools_scm import get_version as scm_get_version
        version = scm_get_version(
            root='..',
            relative_to=__file__,
            version_scheme=version_scheme,
            local_scheme=local_scheme
        )
    except ImportError:
        try:
            try:
                # this works in Python 3.8
                from importlib.metadata import PackageNotFoundError, \
                    version as importlib_version
            except ImportError:
                # this works in Python 2 if our package is installed, since
                # it depends on importlib_metadata
                from importlib_metadata import PackageNotFoundError, \
                    version as importlib_version
            try:
                # we land here if either importlib.metadata or
                # importlib_metadata is available
                # we expect that this module is a submodule of our actual
                # package
                # we expect that __name__ is <our_package>._version
                # therefore, splitting on . should allow us to get the name of
                # our package
                version = importlib_version(__name__.split('.')[0])
            except PackageNotFoundError:
                # we will land here if our package isn't actually installed
                version = 'unknown'
        except ImportError:
            # we land here if neither importlib.metadata nor importlib_metadata
            # are available
            version = 'unknown'
    return version


def main():
    print(get_version())


if __name__ == '__main__':
    main()
