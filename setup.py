from setuptools import setup, find_packages
import os


def read(fname):
    fpath = os.path.join(os.path.dirname(__file__), fname)
    return open(fpath).read()


def find_version_from_readme():
    s = read('README.md')
    i0 = s.lower().find('version')
    i1 = i0 + 20
    v = s[i0:i1].splitlines()[0]  # removes next line
    v = v.split(' ')[1]  # finds version number
    return v


def walker(base, *paths):
    file_list = set([])
    cur_dir = os.path.abspath(os.curdir)

    os.chdir(base)
    try:
        for path in paths:
            for dname, _, files in os.walk(path):
                for f in files:
                    file_list.add(os.path.join(dname, f))
    finally:
        os.chdir(cur_dir)

    return list(file_list)


setup(
    # Application name:
    name='phyto_photo_utils',

    # Version number (initial):
    version=find_version_from_readme(),

    # Application author details:
    author='Thomas Ryan-Keogh, Charlotte Robinson',
    author_email='tjryankeogh@gmail.com',

    # Packages
    packages=find_packages(),

    # Include additional files into the package
    include_package_data=True,

    # Details
    url='https://gitlab.com/tjryankeogh/phytophotoutils',
    download_url= 'https://gitlab.com/tjryankeogh/phytophotoutils/-/archive/v1.3.3/phytophotoutils-v1.3.3.tar.gz',
    license='MIT License',
    description='Tools and utilities for active chlorophyll fluorescence data processing.',
    organisation='Council for Scientific and Industrial Research, Curtin University',

    long_description=read('README.md'),
    long_description_content_type='text/markdown',

    # Dependent packages (distributions)
    install_requires=[
        'tqdm',
        'scipy',
        'numpy',
        'pandas',
        'datetime',
        'matplotlib',
        'sklearn'],
)
