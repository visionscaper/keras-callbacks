from setuptools import setup, find_packages

setup(
    name='keras_callbacks',
    version='0.3',
    packages=find_packages(),
    license='MIT license',
    long_description=open('README.md').read(),
    dependency_links=['git+https://github.com/visionscaper/pybase.git'],
    install_requires=[])
