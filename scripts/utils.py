# 
# modified from: 
#   https://github.com/nv-tlabs/ATISS.
#

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json
import time
import string
import os
import random
import subprocess
import torch

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PATH_TO_DATASET_FILES = os.path.join(PROJ_DIR, "../ThreedFront/dataset_files/")
PATH_TO_PROCESSED_DATA = os.path.join(PROJ_DIR, "../ThreedFront/output/3d_front_processed/")


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def update_data_file_paths(config_data):
    config_data["dataset_directory"] = \
        os.path.join(PATH_TO_PROCESSED_DATA, config_data["dataset_directory"])
    config_data["annotation_file"] = \
        os.path.join(PATH_TO_DATASET_FILES, config_data["annotation_file"])
    return config_data


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_experiment_params(args, experiment_tag, path_to_params):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(path_to_params, "w") as f:
        json.dump(params, f, indent=4)


def get_time_str(seconds):
    hms_str = time.strftime("%H:%M:%S", time.gmtime(seconds))
    if seconds < 24 * 3600:
        return hms_str
    
    day_str = f"{int(seconds // (24 * 3600))} day "
    return day_str + hms_str


def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return None
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}".format(max_id)
    )
    opt_path = os.path.join(
        experiment_directory, "opt_{:05d}".format(max_id)
    )
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return None

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id

    params_path = os.path.join(experiment_directory, "params.json")
    if os.path.exists(params_path) and args.with_wandb_logger:
        wandb_id = json.load(open(params_path, "r")).get("with_wandb_logger")
        return wandb_id
    else:
        return None


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )
