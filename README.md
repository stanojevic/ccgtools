# CCGtools

This is a collection of Python and Cython tools for processing CCG. It contains a pretrained parser, efficient loading of trees, A* search, latex style files, visualization, extraction of semantic dependencies, evaluation scripts etc. It is not very polished but you may find it useful. Probably the most interesting part to play with is a colab notebook for interactive parsing and visualiztion (see below).

## Installation

To install the full parser run

    pip install "ccgtools[parser]@git+https://github.com/stanojevic/ccgtools"

To install just the tools for building and evaluating parsers run

    pip install "ccgtools@git+https://github.com/stanojevic/ccgtools"

All pretrained models have preffix "pretrained:" and will be automatically downloaded when needed.
The list of available models is available [here](https://github.com/stanojevic/ccgtools/blob/main/ccg/supertagger/configs/pretrained_models_locations.tsv).

## Command Line Usage

After the pip line above is exectued there are several commands that will be available on the PATH:
`ccg-eval`, `ccg-parser`, `ccg-supertagger` and `ccg-train`.
If you run them with `--help` flag they will provide the usage directions.

## Notebook

You can [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stanojevic/ccgtools/blob/master/notebooks/demo.ipynb)
the provided notebook and start playing with a CCG parser. It can take a couple of minutes for the Colab to install all the necessary dependencies, but after that is done everything should be fast.

## Author
Miloš Stanojević
milosh.stanojevic@gmail.com
