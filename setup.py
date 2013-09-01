#!/usr/bin/env python
#-*- coding: UTF-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A tool for computing Nash equilibria',
    'author': 'Petr Šebek',
    'license': 'MIT',
    'url': 'https://github.com/Artimi/neng',
    'author_email': 'petrsebek1@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'numpy'],
    'packages': ['neng'],
    'scripts': [],
    'name': 'neng',
    'entry_points': {
        'console_scripts': [
            'neng = neng.neng:main'
        ]
    }
}

setup(**config)
