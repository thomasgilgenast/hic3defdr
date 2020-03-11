"""
Module for managing version naming scheme and supplying version information.

The version naming scheme is specified by defining functions version_scheme()
and local_scheme(). We have written these so that they mimic versioneer's
"pep440" style. These functions are used in this module, and they are also
imported in ``setup.py`` at install time.

When hic3defdr is installed, its version can be accessed as
``hic3defdr.__version__``. This is the recommended way to get version
information when it is known that hic3defdr is installed.

This script provides an alternative way to access the version that will work if
either hic3defdr or setuptools_scm is installed. This is primarily useful for
Docker image building, where it doesn't make sense to make the host install
hic3defdr just to obtain the version information to pass to the image build
process.

Call ``get_version()`` to obtain the version information, or run this file as a
script to print version information to stdout.
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
        # if hic3defdr is installed this should work
        from hic3defdr import __version__ as version
        assert version != 'unknown'
    except (ImportError, AssertionError):
        try:
            from setuptools_scm import get_version as scm_get_version
            version = scm_get_version(
                root='..',
                relative_to=__file__,
                version_scheme=version_scheme,
                local_scheme=local_scheme
            )
        except ImportError:
            raise ValueError('please install either hic3defdr or '
                             'setuptools_scm')
    return version


def main():
    print(get_version())


if __name__ == '__main__':
    main()
