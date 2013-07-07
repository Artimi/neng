#!/usr/bin/env python
#-*- coding: UTF-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A tool for computing Nash equilibria',
    'author': 'Petr Šebek',
    #'url': 'URL to get it at.',
    #'download_url': 'Where to download it.',
    'author_email': 'petrsebek1@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'scipy'],
    'packages': ['neng'],
    'scripts': [],
    'name': 'neng'
}

setup(**config)
