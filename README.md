# CCGtools

Don't play with this code yet. NOT READY

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
the provided notebook and start playing with a CCG parser.

## Author
Miloš Stanojević
milosh.stanojevic@gmail.com
