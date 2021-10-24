from setuptools import setup, find_packages
from Cython.Build import cythonize
from glob import glob
import os

extensions = cythonize("ccg/**/*.pyx", language_level=3)

with open('requirements.txt') as fh:
    install_requires = [x.strip() for x in fh]

with open('requirements-parser.txt') as fh:
    install_requires_parser = [x.strip() for x in fh]

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
    package_data={'': glob("ccg/*.txt")+glob("ccg/*.jar")+glob("ccg/*.sty")+glob("ccg/supertagger/configs/*.yaml")},
    include_package_data=True,
    install_requires=install_requires,
    extras_require={"parser": install_requires_parser},
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'ccg-split = ccg:_main_split',
            'ccg-eval = ccg.evaluation:_main_evaluate',
            'ccg-parser = ccg.supertagger.main:parse',
            'ccg-supertagger = ccg.supertagger.main:supertagger',
            'ccg-train = ccg.supertagger.main:train',
        ]
    },
    classifiers=[
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
                ],
)
