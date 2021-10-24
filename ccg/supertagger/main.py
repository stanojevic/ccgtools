import argparse
import yaml


def _load_and_setup_model(model_name, gpu_id):
    from ccg.supertagger.parser import Parser
    parser = Parser(model_name)
    parser.to(gpu_id)
    return parser


def supertagger():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model"                      , required=True)
    arg_parser.add_argument("--input-text-fn"              , required=True)
    arg_parser.add_argument("--output-tags-fn"             , required=True)
    arg_parser.add_argument("--multitaggin-top-k", type=int, default=0    )
    arg_parser.add_argument("--gpu_id",            type=int, default=None )
    args = arg_parser.parse_args()
    parser = _load_and_setup_model(args.model, args.gpu_id)
    with open(args.input_text_fn) as fh_in, \
            open(args.output_tags_fn, "w") as fh_out:
        raise NotImplementedError()
        # for tree in parser.stag_iter(fh_in):
        #     raise NotImplementedError()


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


def parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model"                 , required=True      )
    arg_parser.add_argument("--input-text-fn"         , required=True      )
    arg_parser.add_argument("--output-trees-fn"       , required=True      )
    arg_parser.add_argument("--gpu_id",       type=int, default=None       )
    arg_parser.add_argument("--num-cpus",     type=int, default=None       )
    args = arg_parser.parse_args()

    parser = _load_and_setup_model(args.model, args.gpu_id)
    with open(args.input_text_fn) as fh_in, \
         open(args.output_trees_fn, "w") as fh_out:
        for tree in parser.parse_iter(fh_in):
            print(tree.to_ccgbank_str(), file=fh_out)
            print(stags)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trees-file", default=None)
    parser.add_argument("--train-stags-file", default=None)
    parser.add_argument("--dev-trees-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--min-epochs", type=int, required=True)
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()
    assert args.train_trees_file is not None or args.train_stags_fle is not None, "training file must be provided"
    args.gpus = eval(args.gpus)

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
              max_epochs=args.max_epochs)


if __name__ == "__main__":
    # parse()
    train()
