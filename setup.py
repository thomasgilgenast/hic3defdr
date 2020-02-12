from setuptools import setup, find_packages

import versioneer

with open('README.md') as fobj:
    long_description = fobj.read()

extras_require = {
    'evaluation': ['scikit-learn>=0.20.3'],
    'test': ['nose>=1.3.7', 'nose-exclude>=0.5.0', 'flake8>=3.4.1'],
    'progress': ['tqdm>=4.32.2'],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='hic3defdr',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='a genome-scale differential loop finder',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Thomas Gilgenast',
    url='https://bitbucket.org/creminslab/hic3defdr',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.2.0',
        'matplotlib>=2.1.1',
        'mpl_scatter_density>=0.6',
        'seaborn>=0.8.0',
        'pandas>=0.21.0',
        'lib5c>=0.5.3',
        'dill>=0.2.9',
    ],
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
