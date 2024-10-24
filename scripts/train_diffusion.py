# 
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
# 

"""Script used to train a diffusion models."""
import argparse
import os
import sys
import shutil
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import PROJ_DIR, update_data_file_paths, id_generator, save_experiment_params, \
    load_config, get_time_str, load_checkpoints, save_checkpoints
from midiffusion.datasets.threed_front_encoding import get_encoded_dataset
from midiffusion.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from midiffusion.stats_logger import StatsLogger, WandB


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR+"/output/log",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
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

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(args.output_directory, experiment_tag)
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Parse the config file
    config = load_config(args.config_file)
    shutil.copyfile(
        args.config_file, os.path.join(experiment_directory, "config.yaml")
    )

    train_dataset = get_encoded_dataset(
        update_data_file_paths(config["data"]),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=(config["network"]["room_mask_condition"] and \
                           config["feature_extractor"]["name"]=="resnet18")
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"],
        #add objfeats
        objfeats=train_dataset.bounds["objfeats"],
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    validation_dataset = get_encoded_dataset(
        update_data_file_paths(config["data"]),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=(config["network"]["room_mask_condition"] and \
                           config["feature_extractor"]["name"]=="resnet18")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    print("Training set has {} bounds".format(list(train_dataset.bounds.keys())))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    print("Validation set has {} bounds".format(list(validation_dataset.bounds.keys())))

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.n_object_types, config, args.weight_file, device=device
    )
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in \
        filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}: "
          f"{n_trainable_params} / {n_all_params}")
    config["network"]["n_params"] = n_trainable_params

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], \
        filter(lambda p: p.requires_grad, network.parameters())) 
    # optimizer = optimizer_factory(config["training"], network.parameters() )

    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Load the learning rate scheduler 
    lr_scheduler = schedule_factory(config["training"])

    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get("project", "MiDiffusion"),
            name=experiment_tag,
            watch=False,
            log_frequency=10
        )
        args.with_wandb_logger = WandB.instance().id

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"), "w"
    ))

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))

    # Do the training
    epochs = config["training"].get("epochs", 150)
    max_grad_norm = config["training"].get("max_grad_norm", None)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    min_val_loss = float("inf")
    min_val_loss_epoch = 0
    tic = time.perf_counter()
    for i in range(args.continue_from_epoch + 1, epochs + 1):
        # adjust learning rate
        adjust_learning_rate(lr_scheduler, optimizer, i)

        network.train()
        #for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
        for b, sample in enumerate(train_loader):
            # Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            batch_loss = train_on_batch(network, optimizer, sample, max_grad_norm)
            StatsLogger.instance().print_progress(i, b+1, batch_loss)

        if (i % save_every) == 0:
            save_checkpoints(i, network, optimizer, experiment_directory)
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            val_loss_total = 0.0
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if not isinstance(v, list):
                        sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample)
                StatsLogger.instance().print_progress(-1, b+1, batch_loss)
                val_loss_total += batch_loss
            StatsLogger.instance().clear()

            toc = time.perf_counter()
            elapsed_time = toc - tic
            estimated_total_time = (elapsed_time / (i - args.continue_from_epoch))\
                * (epochs - args.continue_from_epoch)
            print("====> [Elapsed time: {}] / [Estimated total: {}] ====>".format(
                get_time_str(elapsed_time), get_time_str(estimated_total_time)
            ))
            
            if val_loss_total < min_val_loss:
                # Overwrite best_model.pt
                min_val_loss = val_loss_total
                min_val_loss_epoch = i
                torch.save(network.state_dict(),
                           os.path.join(experiment_directory, "best_model.pt"))
            
    print("Best model saved at epcho {} with validation loss = {}".format(
        min_val_loss_epoch, min_val_loss), 
        file=open(os.path.join(experiment_directory, "stats.txt"), "a")
    )


if __name__ == "__main__":
    main(sys.argv[1:])
