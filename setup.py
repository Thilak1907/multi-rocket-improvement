import os
import codecs
from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


requirements = []
with open('requirements.txt') as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)


setup(
    name='MultiRocket',
    author='Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey Webb',
    author_email='ChangWeiTan',
    license='GNU General Public License v3.0',
    url='https://github.com/Thilak1907/multi-rocket-improvement',
    description='''MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification''',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['multirocket']),
    python_requires='>=3.11',
    install_requires=requirements,
    version=get_version("multirocket/__init__.py"),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: General Public License',
    ],
)
