from ccg.derivation import Node, DerivationLoader
from ccg.categories import Category
from ccg import combinators as comb
from ccg.penn_treebank_tokenizer import penn_tokenize
from os import path
import os

import jieba
import logging
logging.getLogger("jieba").setLevel(logging.CRITICAL)


_original_open = open
open = DerivationLoader.iter_from_file


def open_treebank_dir(dir_name):
    dl = DerivationLoader
    fs = dl.iter_from_treebank_train, dl.iter_from_treebank_dev, dl.iter_from_treebank_test, dl.iter_from_treebank_rest
    return tuple(f(dir_name) for f in fs)


def parser(parser_name,
           words_per_batch : int = 25*100,
           do_tokenization: bool = True,
           max_steps: int = 1_000_000,
           prune_beta: float = 0.0001,
           prune_top_k_tags: int = 50,
           num_cpus : int = None):
    from ccg.supertagger.parser import Parser
    parser = Parser(parser_name,
                    words_per_batch=words_per_batch,
                    do_tokenization=do_tokenization,
                    max_steps=max_steps,
                    prune_beta=prune_beta,
                    prune_top_k_tags=prune_top_k_tags,
                    num_cpus=num_cpus)
    return parser


def tokenize(sent, language):
    if language in ["Chinese", "zh"]:
        if type(sent) == list:
            sent = " ".join(sent)
        return list(jieba.cut(sent, cut_all=False))
    elif language in ["English", "en"]:
        return penn_tokenize(sent, convert_parentheses=False)
    else:
        raise Exception(f"unsupported tokenization for {language} language")


derivation = DerivationLoader.from_str
category = Category.from_str

TypeChanging1 = comb.TypeChanging1  # class
TypeRaising = comb.TypeRaising  # class
tr_np_fwd = TypeRaising(cat_res=category("S"), cat_arg=category("NP"), is_forward=True)
tr_np_bck = TypeRaising(cat_res=category("S\\NP"), cat_arg=category("NP"), is_forward=False)

TypeChanging2 = comb.TypeChanging2  # class
RightAdjoin = comb.RightAdjoin  # class

Glue = comb.Glue  # class
glue = comb.Glue()

Conj = comb.Conj  # class
up_conj = comb.Conj(is_bottom=False)
bottom_conj = comb.Conj(is_bottom=True)

Punc = comb.Punc  # class
lpunc = comb.Punc(punc_is_left=True)
rpunc = comb.Punc(punc_is_left=False)

B = comb.B  # class
S = comb.S  # class

bx1f = B(is_forward=True, is_crossed=True, order=1)
bx1b = B(is_forward=False, is_crossed=True, order=1)
b1f = B(is_forward=True, is_crossed=False, order=1)
b0f = B(is_forward=True, is_crossed=False, order=0)
b0b = B(is_forward=False, is_crossed=False, order=0)

fapply = b0f
bapply = b0b
fcomp = b1f
fxcomp = bx1f


def _main_split():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--treebank-dir", required=True)
    parser.add_argument("--output-dir"  , required=True)
    args = parser.parse_args()

    train, dev, test, rest = open_treebank_dir(args.treebank_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    for s, trees in [('train', train), ('dev', dev), ('test', test), ('rest', rest)]:
        print(f"processing {s}")
        fn_ccg = path.join(args.output_dir, s + ".auto")
        fn_stag = path.join(args.output_dir, s + ".stags")
        fn_words = path.join(args.output_dir, s + ".words")
        assert not path.exists(fn_ccg), f"file {fn_ccg} must be removed before proceeding"
        with _original_open(fn_ccg, "w") as fh_ccg, \
             _original_open(fn_stag, "w") as fh_stag, \
             _original_open(fn_words, "w") as fh_words:
            for tree in trees:
                print(tree.to_ccgbank_str(), file=fh_ccg)
                print(" ".join(map(str, tree.stags())), file=fh_stag)
                print(" ".join(tree.words()), file=fh_words)