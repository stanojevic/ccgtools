from setuptools import setup, find_packages
from glob import glob
import os

USE_CYTHON = True  # os.environ.get('USE_CYTHON', "False") in ["True", "1"]


if USE_CYTHON:
    try:
        from Cython.Build import cythonize
    except ImportError:
        sys.exit("\n\n\tCould not import Cython, which is required to build ccgtools.\n\n")
    extensions = cythonize("ccg/**/*.pyx", language_level=3)
else:
    from setuptools import Extension
    extensions = [Extension(x[:-2].replace("/", "."), [x]) for x in glob("ccg/*.c")]

with open('requirements.txt') as fh:
    install_requires = [x.strip() for x in fh]


with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()


setup(
    name="ccgtools",
    author="Miloš Stanojević",
    version="0.0.1",
    description='Tools for working with Combinatory Categorial Grammar (CCG)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=extensions,
    url="https://github.com/stanojevic/ccgtools",
    license='MIT License',
    packages=find_packages(),
    package_data={'': glob("ccg/*.txt")+glob("ccg/*.jar")+glob("ccg/*.sty")},
    include_package_data=True,
    install_requires=install_requires,
    zip_safe=False,
    classifiers=[
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
                ],
)
