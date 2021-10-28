import json
import sys
import os
import argparse
import time

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.params import with_fallback

from allennlp.common.util import import_module_and_submodules


args = argparse.ArgumentParser()
args.add_argument('name', nargs='?', default='')

args.add_argument('--comp-split', type=str)
args.add_argument('--few-shot-examples', type=int, default=0)
args.add_argument('--text-baseline', action='store_true')
args.add_argument('--volta-path', default='pretrained_volta/volta_models')
args.add_argument('--volta-model', default='ctrl_visualbert_base')

args.add_argument('--force', action='store_true')
args.add_argument('--recover', action='store_true')
args.add_argument("-o", "--overrides", type=str, default="")
args = args.parse_args()

if __name__ == "__main__":
    import_module_and_submodules('src')

    parameter_filename = "train_configs/defaults.jsonnet"
    settings = Params.from_file(parameter_filename, args.overrides).params

    if args.text_baseline:
        text_baseline_parameter_filename = "train_configs/text_baseline.jsonnet"
        text_baseline_settings = Params.from_file(text_baseline_parameter_filename, args.overrides).params
        settings = with_fallback(text_baseline_settings, settings)

    settings['model']['pretrained_model_path'] = os.path.join(args.volta_path, args.volta_model)
    settings['model']['pretrained_model_config'] = os.path.join("pretrained_volta/volta_configs", f"{args.volta_model}.json")

    pretrained_config = json.load(open(settings['model']['pretrained_model_config']))
    settings['dataset_reader']['num_vis_position_features'] = pretrained_config['num_locs']

    experiment_name = "iid"
    if args.comp_split:
        experiment_name = args.comp_split
        settings['dataset_reader']['comp_split'] = args.comp_split
        settings['dataset_reader']['few_shot_examples'] = args.few_shot_examples

    # check if in pycharm debug mode - in such case remove multiprocessing (there are some issues with that)
    debug_mode = sys.gettrace() is not None
    if debug_mode:
        force = True
        settings['data_loader']['num_workers'] = 0
        settings['validation_data_loader']['num_workers'] = 0

    parent_experiments_path = "runs"
    serialization_dir_path = os.path.join(parent_experiments_path, experiment_name)

    if not debug_mode:
        if os.path.exists(serialization_dir_path):
            serialization_dir = experiment_name + '_' + str(time.time()).replace('.', '')
            serialization_dir_path = os.path.join(parent_experiments_path, serialization_dir)

    train_model(
        params=Params(settings),
        serialization_dir=serialization_dir_path,
        recover=args.recover,
        force=args.force
    )
