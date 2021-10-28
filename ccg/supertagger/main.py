import argparse
import yaml
from os.path import join, basename
from sys import stderr
import numpy as np


def _load_and_setup_model(args, parsing=True):
    from ccg.supertagger.parser import Parser

    if parsing:
        max_steps = args.max_steps
        prune_beta = args.prune_beta
        num_cpus = args.num_cpus
        prune_top_k_tags = args.prune_top_k_tags
    else:
        max_steps = 10_000_000
        prune_beta = 0.0000001
        num_cpus = 1
        prune_top_k_tags = 0

    parser = Parser(args.model,
                    words_per_batch=args.words_per_batch,
                    do_tokenization=args.do_tokenization,
                    max_steps=max_steps,
                    prune_beta=prune_beta,
                    prune_top_k_tags=prune_top_k_tags,
                    num_cpus=num_cpus)
    parser.to(args.gpu_id)
    return parser


def _add_parser_args(args, parsing=True):
    args.add_argument("--model"                        , required=True )
    args.add_argument("--words-per-batch",   type=int  , default=25*100)
    args.add_argument("--no-tokenization", dest='do_tokenization', default=True, action='store_false')
    args.add_argument("--gpu-id",            type=int  , default=None  )
    if parsing:
        args.add_argument("--max-steps",         type=int  , default=10_000_000)
        args.add_argument("--prune-beta",        type=float, default=0.0001)
        args.add_argument("--prune-top-k-tags",  type=int  , default=50)
        args.add_argument("--num-cpus",          type=int  , default=0     )
    else:
        args.add_argument("--multitagging-top-k", type=int, default=None)


def supertagger():
    arg_parser = argparse.ArgumentParser()
    _add_parser_args(arg_parser, parsing=False)
    arg_parser.add_argument("--input-text-fn" , required=True)
    arg_parser.add_argument("--output-tags-fn", required=True)
    args = arg_parser.parse_args()

    top_k = args.multitagging_top_k
    is_multitagging = top_k is not None and top_k > 0
    parser = _load_and_setup_model(args, parsing=False)
    with open(args.input_text_fn) as fh_in, \
            open(args.output_tags_fn, "w") as fh_out:
        for sent_id, stag_logprobs in enumerate(parser.stag_iter(fh_in)):
            if (sent_id+1) % 1000 == 0:
                print(f"processed {sent_id}", file=stderr)
            ls, ts = stag_logprobs.shape
            if is_multitagging:
                tags_total = stag_logprobs.shape[-1]
                k = min(tags_total, top_k)
                stag_indices = np.argpartition(stag_logprobs, -k, axis=-1)[:, -k:]
                stag_logprobs = np.take_along_axis(stag_logprobs, stag_indices, axis=-1)
                for i in range(ls):
                    out_strs = []
                    for j in range(k):
                        out_strs.append(f"{parser.model.stag2i.i2s(stag_indices[i, j])}={stag_logprobs[i, j]}")
                    print(" ".join(out_strs), file=fh_out)
                print(file=fh_out)
            else:
                best_indices = np.argmax(stag_logprobs, axis=-1).tolist()
                best_stags = [parser.model.stag2i.i2s(ind) for ind in best_indices]
                print(" ".join(map(str, best_stags)), file=fh_out)


def parse():
    arg_parser = argparse.ArgumentParser()
    _add_parser_args(arg_parser, parsing=True)
    arg_parser.add_argument("--input-text-fn"  , required=True)
    arg_parser.add_argument("--output-trees-fn", required=True)
    args = arg_parser.parse_args()

    parser = _load_and_setup_model(args, parsing=True)
    with open(args.input_text_fn) as fh_in, \
         open(args.output_trees_fn, "w") as fh_out:
        for sent_id, tree in enumerate(parser.parse_iter(fh_in)):
            if (sent_id+1) % 1000 == 0:
                print(f"processed {sent_id} sentences", file=stderr)
            print(tree.to_ccgbank_str(), file=fh_out)


def _load_config(config_name):
    from os.path import realpath, dirname, join, isfile
    from os import getcwd
    if not isfile(config_name):
        config_name2 = join(realpath(join(getcwd(), dirname(__file__))), "configs", config_name+".yaml")
        if not isfile(config_name2):
            raise Exception(f"unknown config file {config_name}")
        config_name = config_name2
    with open(config_name) as fh:
        params = yaml.safe_load(fh)
    return params


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trees-file", default=None)
    parser.add_argument("--train-stags-file", default=None)
    parser.add_argument("--dev-trees-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--gpus", type=str, default="None")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    assert args.train_trees_file is not None or args.train_stags_fle is not None, "training file must be provided"
    args.gpus = eval(args.gpus)
    if args.save_dir is None:
        if "/" not in args.config and "\\" not in args.config:
            args.save_dir = join(".", "models", args.config)
        else:
            raise Exception("you have to specify the --save_dir argument")
    if args.language != basename(args.config).split("-")[0]:
        raise Exception("requested language and config file don't agree")
    params = _load_config(args.config)
    from optuna.trial import FixedTrial
    trial = FixedTrial(params)
    from ccg.supertagger.dataset import prepare_data
    prepared = prepare_data(args.train_trees_file, args.train_stags_file, args.dev_trees_file)
    from ccg.supertagger.model import objective
    objective(trial=trial,
              save_dir=args.save_dir,
              train_dataset=prepared['train_dataset'],
              dev_dataset=prepared['dev_dataset'],
              w2i=prepared['w2i'],
              stag2i=prepared['stag2i'],
              gpus=args.gpus,
              seed=args.seed,
              min_epochs=args.min_epochs,
              max_epochs=args.max_epochs,
              language=args.language)
