from distutils.core import setup

setup(
    name='keras_callbacks',
    version='0.1',
    packages=['keras_callbacks'],
    license='MIT license',
    long_description=open('README.md').read(),
    install_requires=[
        'keras>=2.1.6',
        'git+https://github.com/visionscaper/pybase.git'
    ],
)