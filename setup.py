#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Medical_Segmentation_Benchmark',
    version='0.0.1',
    description='A medical image segmentation project for open dataset by using pytorch project',
    author='Jia-Ming Hou',
    author_email='t110368027@ntut.org.tw',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/t110368027/Medical_Segmentation_Benchmark',
    install_requires=['pytorch-lightning','albumentations','monai'],
    packages=find_packages(),
)

