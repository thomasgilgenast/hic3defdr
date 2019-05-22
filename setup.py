from setuptools import setup, find_packages

import versioneer

readme_note = """\
.. note::
   For the latest source, discussion, etc, please visit the
   `Bitbucket repository <https://bitbucket.org/creminslab/fast3defdr>`_\n\n
"""

with open('README.md') as fobj:
    long_description = readme_note + fobj.read()

setup(
    name='fast3defdr',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='a genome-scale differential loop finder',
    long_description=long_description,
    author='Thomas Gilgenast',
    url='https://bitbucket.org/creminslab/fast3defdr',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.2.0',
        'matplotlib>=2.1.1',
        'seaborn>=0.8.0',
        'pandas>=0.21.0',
        'lib5c>=0.5.3',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
