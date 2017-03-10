#!/usr/bin/env python
from setuptools import setup

setup(
    name='idsim',
    author='Daniel Kappler, Franziska Meier',
    author_email='daniel.kappler@gmail.com, fmeier@gmail.com',
    version=1.0,
    packages=['inverse_dynamics'],
    package_dir={'inverse_dynamics': ''},
    zip_safe=False,
)
