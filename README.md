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

One thing to note is that the parser's code that does search is implemented in Java which requires further tweaks for very long sentences that can be memory and computationally intensive. Java by default doesn't reserve much memory so parser can run out of it on long sentences. To give Java Virtual Machine permision to use more memory run `export JAVA_TOOL_OPTIONS="-Xms512m -Xmx12g"`. I have set there the maximum value of RAM to 12 GB, but of course adapt that to your own machine. Even with large memory, the search can be very long if the sentence is very long and very ambiguous syntactically. To prevent parser from searching for a very long time you can set the parameter `--max-steps 1000000` (or some higher value if you are more patient than me).


## Notebook

There are two notebooks included in the repository:

* demo.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stanojevic/ccgtools/blob/master/notebooks/demo.ipynb)
that shows how to use the provided parser, different ways of visualizing derivations, extraction of semantic dependencies, and extraction of logical representation.
It can take a couple of minutes for the Colab to install all the necessary dependencies, but after that is done everything should be fast.
* processing\_effort.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stanojevic/ccgtools/blob/master/notebooks/processing_effort.ipynb) computes processing effort predictions for different CCG derivation trees. This follows descriptions from the papers that relate CCG parsing effort to brain activity ([paper 1](https://aclanthology.org/2021.cmcl-1.3.pdf) and [paper 2](https://arxiv.org/abs/2210.16147)).

## Author
Miloš Stanojević
milosh.stanojevic@gmail.com
