#!/usr/bin/env python

# cd alphapy-pro
# python setup.py sdist bdist_wheel
# pip install -e .

from setuptools import find_packages
from setuptools import setup
import re
import os

# Read version from package init
def get_version():
    init_path = os.path.join('alphapy', '__init__.py')
    with open(init_path, 'r') as f:
        content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

DISTNAME = 'alphapy-pro'
DESCRIPTION = "AlphaPy: A Machine Learning Pipeline for Speculators"
LONG_DESCRIPTION = "alphapy is a Python library for machine learning using scikit-learn. We have a stock market pipeline and a sports pipeline so that speculators can test predictive models, along with functions for trading systems and portfolio management."

MAINTAINER = 'ScottFree LLC [Robert D. Scott II, Mark Conway]'
MAINTAINER_EMAIL = 'scottfree.analytics@scottfreellc.com'
URL = "https://github.com/ScottFreeLLC/alphapy-pro"
LICENSE = "Apache License, Version 2"
VERSION = get_version()

classifiers = ['Development Status :: 4 - Beta',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.12',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

install_reqs = [
    'arrow>=0.13',
    'bokeh>=1.3',
    'category_encoders>=2.1',
    'iexfinance>=0.4.3',
    'imbalanced-learn>=0.5',
    'ipython>=7.2',
    'lofo-importance>=0.3.4',
    'matplotlib>=3.0',
    'numpy>=1.17',
    'pandas>=1.0',
    'pandas-datareader>=0.8',
    'polygon-api-client>=1.13.4',
    'pyyaml>=5.0',
    'scikit-learn>=0.23.1',
    'scipy>=1.5',
    'seaborn>=0.9',
    'venn-abers>=1.4.1',
    'yfinance>=0.1.59',
]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        classifiers=classifiers,
        install_requires=install_reqs,
        entry_points={
            'console_scripts': [
                'alphapy = alphapy.alphapy_main:main',
                'mflow = alphapy.mflow_main:main',
            ],
        }
    )
