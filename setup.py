from setuptools import setup, find_packages

from hic3defdr._version import version_scheme, local_scheme

with open('README.md') as fobj:
    long_description = fobj.read()

extras_require = {
    'evaluation': ['scikit-learn>=0.20.3'],
    'test': [
        'nose>=1.3.7',
        'nose-exclude>=0.5.0',
        'doctest-ignore-unicode>=0.1.2',
        'flake8>=3.4.1',
    ],
    'progress': ['tqdm>=4.32.2'],
    'docs': [
        'Sphinx>=1.8.5',
        'sphinx-rtd-theme>=0.4.3',
        'sphinxcontrib-apidoc>=0.3.0',
        'm2r>=0.2.1',
    ]
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='hic3defdr',
    use_scm_version={
        'version_scheme': version_scheme,
        'local_scheme': local_scheme,
    },
    description='a genome-scale differential loop finder',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Thomas Gilgenast',
    author_email='thomasgilgenast@gmail.com',
    url='https://bitbucket.org/creminslab/hic3defdr',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.2.0',
        'matplotlib>=2.1.1',
        'mpl_scatter_density>=0.6',
        'seaborn>=0.8.0',
        'pandas>=0.21.0',
        'lib5c>=0.6.0',
        'dill>=0.2.9',
        'importlib_metadata>=1.5.0;python_version<"3.8"',
    ],
    setup_requires=['setuptools_scm'],
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License'
    ],
)
