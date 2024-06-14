"""Script used for generating results using a previously trained model."""
import argparse
import os
import sys
import shutil
import pickle

import numpy as np
import torch

from utils import PROJ_DIR, load_config, update_data_file_paths
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from midiffusion.datasets.threed_front_encoding import get_dataset_raw_and_encoded
from midiffusion.networks import build_network
from midiffusion.evaluation.utils import generate_layouts


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "weight_file",
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to the file that contains the experiment configuration"
        "(default: config.yaml in the model directory)"
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR+"/output/predicted_results/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--n_known_objects",
        default=0,
        type=int,
        help="Number of existing objects for scene completion task"
    )
    parser.add_argument(
        "--experiment",
        default="synthesis",
        choices=[
            "synthesis",
            "scene_completion",
            "furniture_arrangement",
            "object_conditioned",
            "scene_completion_conditioned"
        ],
        help="Experiment name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the sampling floor plan"
    )
    parser.add_argument(
        "--n_syn_scenes",
        default=1000,
        type=int,
        help="Number of scenes to be synthesized"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Number of synthesized scene in each batch"
    )
    parser.add_argument(
        "--result_tag",
        default=None,
        help="Result sub-directory name"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
    )

    args = parser.parse_args(argv)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if args.gpu < torch.cuda.device_count():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if result_dir exists and if it doesn't create it
    if args.result_tag is None:
        result_dir = args.output_directory
    else:
        result_dir = os.path.join(args.output_directory, args.result_tag)
    if os.path.exists(result_dir) and \
        len(os.listdir(result_dir)) > 0:
        input("{} direcotry is non-empty. Press any key to remove all files..." \
              .format(result_dir))
        for fi in os.listdir(result_dir):
            os.remove(os.path.join(result_dir, fi))
    else:
        os.makedirs(result_dir, exist_ok=True)

    # Run control files to save
    path_to_config = os.path.join(result_dir, "config.yaml")
    path_to_results = os.path.join(result_dir, "results.pkl")

    # Parse the config file
    if args.config_file is None:
        args.config_file = os.path.join(os.path.dirname(args.weight_file), "config.yaml")
    config = load_config(args.config_file)
    if "_eval" not in config["data"]["encoding_type"] and args.experiment == "synthesis":
        config["data"]["encoding_type"] += "_eval"
    if "text" in config["data"]["encoding_type"] and "textfix" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] = config["data"]["encoding_type"].replace("text", "textfix")
    if not os.path.exists(path_to_config) or \
        not os.path.samefile(args.config_file, path_to_config):
        shutil.copyfile(args.config_file, path_to_config)

    # Raw training data (for record keeping)
    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"]), 
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True)
    ) 

    # Get Scaled dataset encoding (without data augmentation)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"]),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=config["network"].get("room_mask_condition", True)
    )
    print("Loaded {} scenes with {} object types ({} labels):".format(
        len(encoded_dataset), encoded_dataset.n_object_types, encoded_dataset.n_classes))
    print(encoded_dataset.class_labels)

    # Build network with saved weights
    network, _, _ = build_network(
        encoded_dataset.n_object_types, config, args.weight_file, device=device
    )
    network.eval()

    # Generate final results
    sampled_indices, layout_list = generate_layouts(
        network, encoded_dataset, config, args.n_syn_scenes, "random",
        experiment=args.experiment, num_known_objects=args.n_known_objects, 
        batch_size=args.batch_size, device=device
    )
    
    threed_front_results = ThreedFrontResults(
        raw_train_dataset, raw_dataset, config, sampled_indices, layout_list
    )
    
    pickle.dump(threed_front_results, open(path_to_results, "wb"))
    print("Saved result to:", path_to_results)
    
    kl_divergence = threed_front_results.kl_divergence()
    print("object category kl divergence:", kl_divergence)
           

if __name__ == "__main__":
    main(sys.argv[1:])