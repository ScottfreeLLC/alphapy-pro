
# cd sflow_app
# pip install -e .

from setuptools import find_packages, setup

setup(
    name='sflow_app',
    version='1.0.0',
    description='sflow application',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'sflow = sflow_app.sflow_main:main',
        ],
    },
)
