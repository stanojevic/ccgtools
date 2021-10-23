ccg-astar-lib
===========

This is an improved version of A* search from Mike Lewis' EasyCCG parser.
This code is meant to be used as a search component of some parser that will provide supertag score.
The repository does not contain any trained model nor code for training.
For that see the original EasyCCG repository. <br/>

The improvements are:
* very small self-contained binary (32 K) that contains only EasyCCG core,
* support for multi-threaded decoding of a batch of sentences without mixing their order,
* replaced blocking multi-threading with the non-blocking one,
* support for span scores,
* pasting together partial subtrees in case the search for a complete parse failed,
* removed dependence on external files for loading combinatory rules,
* access from Python via JPype,
* added type-changing unary rules needed for Chinese CCGbank,
* unary rule penalty,
* usage of normal-form is optional now -- some models give better results in that way,
* has an adjustable number of parsing steps before giving up -- prevents slow runtime on very long and ambiguous sentences,
* few more changes to be able to handle sentences of arbitrary length,
* some heuristics for unary rules (no more than 2 unaries in a row),
* explicit conjunction combinators instead of the buggy X\X.

Compilation requires JDK not older than 1.8.<br/>
Build the self-contained jar by running `ant`.

The original version of EasyCCG was written by Mike Lewis.<br/>
The refactoring of its search component here is done by Miloš Stanojević.

This code is licensed under MIT license.
