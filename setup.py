from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    l_d = f.read()

# Get the version.
version: dict = {}
with open("pastastore/version.py") as fp:
    exec(fp.read(), version)
setup(
    name='pastastore',
    version=version['__version__'],
    description='Tools for managing pastas projects',
    long_description=l_d,
    long_description_content_type='text/markdown',
    url='https://github.com/pastas/pastastore',
    author='D.A. Brakenhoff',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    platforms='Windows, MacOS, *nix',
    install_requires=["numpy>=1.15",
                      "pandas>=1.0,<1.2.0",  # for now, until Arctic is updated
                      "tqdm>=4.36",
                      "pastas>=0.13"],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
