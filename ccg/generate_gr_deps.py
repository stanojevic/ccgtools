import ccg
from ccg.dependencies import DepLink
from ccg.evaluation import sufficient_stats_from_deps, combine_stats
import tempfile
import subprocess
from os.path import realpath, dirname, join
from os import getcwd
import os
import re
from pathlib import Path
from sys import stderr
import gdown
import tarfile


CANDC_WEB_LOCATION = "https://drive.google.com/uc?id=1EUU3fXv7O-618UO71YKot5wkQ-HBHOqN"
CANDC_PARENT = os.path.join(str(Path.home()), ".cache")
CANDC = os.path.join(CANDC_PARENT, "candc-1.00")
GENERATE = join(CANDC, "bin", "generate")
CATS = join(CANDC, "src", "data", "ccg", "cats")
MARKEDUP = join(CATS, "markedup")


def _check_candc_is_installed():
    if os.path.exists(CANDC):
        return
    print("C&C parser is not installed", file=stderr)
    print("downloading and installing now", file=stderr)
    print(f"from {CANDC_WEB_LOCATION}", file=stderr)
    print(f"into {CANDC}", file=stderr)

    gdown.download(CANDC_WEB_LOCATION,  CANDC+".tar.gz")

    tar = tarfile.open(CANDC+".tar.gz", "r:gz")
    tar.extractall(path=CANDC_PARENT)
    tar.close()

    os.unlink(CANDC+".tar.gz")
    os.makedirs(join(CANDC, "bin"))

    for f in ["pool.h", "affix.h", "prob.h", "utils/aux_strings.h"]:
        fn = join(CANDC, "src", "include", f)
        with open(fn) as fh:
            content = fh.read()
        with open(fn, "w") as fh:
            print("#include <string.h>", file=fh)
            print(content, file=fh)

    fn = join(CANDC, "src", "include", "hashtable", "word.h")
    with open(fn) as fh:
        content = fh.read()
    content = content.replace("return hash == this->hash && str == this->str",
                              "return hash == this->hash && str == this->str.str()")
    with open(fn, "w") as fh:
        print(content, file=fh)

    from sys import platform
    if platform.startswith("win"):
        makefile = "Makefile.mingw"
    elif platform.startswith("darwin"):
        makefile = "Makefile.macosxu"
    elif platform.startswith("linux"):
        makefile = "Makefile.unix"
    else:
        raise Exception("unknown system")

    print("starting compilation" , file=stderr)
    install_whole_candc = False
    all = "all" if install_whole_candc else ""
    subprocess.run(f"make -j 8 -f {makefile} {all} bin/generate", shell=True, check=True, cwd=CANDC)
    print("compilation successful !" , file=stderr)


def _tree_convert_to_pipe(node):
    if node.is_term:
        return f"(<L *** {node.cat} X {node.word}>\n)\n"
    elif node.is_binary:
        d = 0 if node.comb.is_B_fwd else 1
        return f"(<T *** {node.cat} * {d} 2>\n{_tree_convert_to_pipe(node.left)}{_tree_convert_to_pipe(node.right)})\n"
    else:
        return f"(<T *** {node.cat} * 0 1>\n{_tree_convert_to_pipe(node.child)})\n"


def _create_pipe_file(trees):
    with tempfile.NamedTemporaryFile(prefix="generate_evaluation_", suffix=".pipe", delete=False, mode="w") as t:
        for tree in trees:
            print("###", file=t)
            print(_tree_convert_to_pipe(tree), file=t)
        return t.name


# g -- grs
# t -- training
# j -- julia_slots -- typically used for evaluation
# T -- text
# e -- ???
def candc_generate_general_form(trees, dep_type="j"):
    _check_candc_is_installed()
    pipe_fn = _create_pipe_file(trees)
    process = subprocess.run([GENERATE, '-'+dep_type, CATS, MARKEDUP, pipe_fn], stdout=subprocess.PIPE)
    os.unlink(pipe_fn)
    all_deps = []
    curr_deps = []
    for line in [x.strip() for x in process.stdout.decode().split("\n")][3:-1]:
        if line:
            curr_deps.append(line)
        else:
            all_deps.append(curr_deps)
            curr_deps = []
    return all_deps


def generate_deps_standard(trees):
    result = []
    for sent_deps in candc_generate_general_form(trees, dep_type="j"):
        deps = set()
        for line in sent_deps:
            pred, cat, slot, arg, rule_id = line.split()[:5]
            pred_word, pred_index = pred.split("_")
            arg_word, arg_index = arg.split("_")
            cat = _strip_markup(cat)
            if not _ignore(pred_word, cat, slot, arg_word, rule_id):
                dep_link = DepLink(head_cat=ccg.category(cat),
                                   head_pos=int(pred_index)-1,
                                   dep_pos=int(arg_index)-1,
                                   dep_slot=int(slot),
                                   head_word=pred_word,
                                   dep_word=arg_word,
                                   is_bound=False,
                                   is_unbound=False,
                                   is_adj=False,
                                   is_conj=False)
                deps.add(dep_link)
        result.append(deps)
    return result


def evaluate(gold_trees, pred_trees):
    gold_deps = [x.deps() for x in gold_trees]
    pred_deps = generate_deps_standard(pred_trees)
    assert len(gold_deps) == len(pred_deps), "the number of trees differ"
    return combine_stats(sufficient_stats_from_deps(g, p) for g, p in zip(gold_deps, pred_deps) if g and p)


_IGNORE_STR = r"""
rule_id 7
rule_id 11
rule_id 12
rule_id 14
rule_id 15
rule_id 16
rule_id 17
rule_id 51
rule_id 52
rule_id 56
rule_id 91
rule_id 92
rule_id 95
rule_id 96
rule_id 98
conj 1 0
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 0
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 2
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 3
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 6
((S[to]{_}\NP{Z}<1>){_}/(S[b]{Y}<2>\NP{Z*}){Y}){_} 1 9
((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 6
((S[b]{_}\NP{Y}<1>){_}/PP{Z}<2>){_} 1 6
(((S[b]{_}\NP{Y}<1>){_}/PP{Z}<2>){_}/NP{W}<3>){_} 1 6
(S[X]{Y}/S[X]{Y}<1>){_} 1 13
(S[X]{Y}/S[X]{Y}<1>){_} 1 5
(S[X]{Y}/S[X]{Y}<1>){_} 1 55
((S[X]{Y}/S[X]{Y}){Z}\(S[X]{Y}/S[X]{Y}){Z}<1>){_} 2 97
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 4
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 93
((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_} 2 8
((S[X]{Y}\NP{Z}){Y}/(S[X]{Y}<1>\NP{Z}){Y}){_} 2 94
((S[X]{Y}\NP{Z}){Y}/(S[X]{Y}<1>\NP{Z}){Y}){_} 2 18
been ((S[pt]{_}\NP{Y}<1>){_}/(S[ng]{Z}<2>\NP{Y*}){Z}){_} 1 0
been ((S[pt]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 there 0
been ((S[pt]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 There 0
be ((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 there 0
be ((S[b]{_}\NP{Y}<1>){_}/NP{Z}<2>){_} 1 There 0
been ((S[pt]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
been ((S[pt]{_}\NP{Y}<1>){_}/(S[adj]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
have ((S[b]{_}\NP{Y}<1>){_}/(S[pt]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[adj]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[ng]{Z}<2>\NP{Y*}){Z}){_} 1 0
be ((S[b]{_}\NP{Y}<1>){_}/(S[pss]{Z}<2>\NP{Y*}){Z}){_} 1 0
going ((S[ng]{_}\NP{Y}<1>){_}/(S[to]{Z}<2>\NP{Y*}){Z}){_} 1 0
have ((S[b]{_}\NP{Y}<1>){_}/(S[to]{Z}<2>\NP{Y*}){Z}){_} 1 0
Here (S[adj]{_}\NP{Y}<1>){_} 1 0
# this is a dependency Julia doesn't have but looks okay
from (((NP{Y}\NP{Y}<1>){_}/(NP{Z}\NP{Z}){W}<3>){_}/NP{V}<2>){_} 1 0
"""
_IGNORE = {tuple(x.strip().split()) for x in _IGNORE_STR.split("\n") if not x.startswith("#")}


def _ignore(pred, cat, slot, arg, rule_id):
    res = ('rule_id', rule_id) in _IGNORE or \
          (cat, slot, rule_id) in _IGNORE or \
          (pred, cat, slot, rule_id) in _IGNORE or \
          (pred, cat, slot, arg, rule_id) in _IGNORE
    return res


_MARKUP = re.compile(r'<[0-9]>|\{[A-Z_]\*?\}|\[X\]')


def _strip_markup(cat):
    cat = _MARKUP.sub('', cat)
    if cat[0] == '(':
        return cat[1:-1]
    else:
        return cat


if __name__ == "__main__":
    AUTO = "/home/milos/PycharmProjects/ccg-tools-private/data/ccgbank_unzipped/data/AUTO/00/wsj_0001.auto"
    gold_trees = list(ccg.open(AUTO))
    with open("/home/milos/proba.gr", "w") as fh:
        for group in candc_generate_general_form(trees=gold_trees, dep_type='g'):
            for line in group:
                print(line, file=fh)
            print(file=fh)