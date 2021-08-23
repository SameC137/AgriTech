#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["pandas==1.3.2","PDAL==2.4.2","geopandas==0.9.0","matplotlib==3.4.3","numpy==1.21.2","pyproj==3.1.0","Shapely==1.7.1"]

test_requirements = ['pytest>=3', ]

setup(
    author="Same Michael",
    email="samemichael1415@gmail.com",
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description="Retrieve 3DEP elevation data, visualize elevation data, transform elevation data",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='pdal, geospatial, elevation, 3DEP, pointcloud, visualization',
    name='elevation_3DEP',
    packages=find_packages(include=['scripts', 'scripts.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/SameC137/AgriTech',
    version='0.1.0',
    zip_safe=False,
)