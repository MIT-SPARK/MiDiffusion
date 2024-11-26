#
# Modified from:
#   https://github.com/tangjiapeng/DiffuScene
#

import os
import math
import torch
from torch.nn.utils import clip_grad_norm_

from .feature_extractors import get_feature_extractor
from .diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from .diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed
from ..stats_logger import StatsLogger


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    # weight_decay = config.get("weight_decay", 0.0)
    # Weight decay was set to 0.0 in the paper's experiments. We note that
    # increasing the weight_decay deteriorates performance.
    weight_decay = 0.0

    if optimizer == "SGD":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        return torch.optim.RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()


def train_on_batch(model, optimizer, sample_params, max_grad_norm=None):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    # Do the backpropagation
    loss.backward()
    # Compute model norm
    if max_grad_norm is not None:
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    StatsLogger.instance()["gradnorm"].value = grad_norm.item()
    # log learning rate
    StatsLogger.instance()["lr"].value = optimizer.param_groups[0]['lr']
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch(model, sample_params):
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    return loss.item()


def build_network(n_object_types, config, weight_file=None, device="cpu"):
    network_type = config["network"]["type"]

    feature_extractor = get_feature_extractor(
        **config["feature_extractor"]
    ) if config["network"].get("room_mask_condition", True) else None

    if network_type == "diffusion_scene_layout_ddpm":
        network = DiffusionSceneLayout_DDPM(
            n_object_types,
            feature_extractor,
            config["network"],
            os.path.join(config["data"]["dataset_directory"], 
                         config["data"]["train_stats"])
        )
    elif network_type == "diffusion_scene_layout_mixed":
        network = DiffusionSceneLayout_Mixed(
            n_object_types,
            feature_extractor,
            config["network"],
            os.path.join(config["data"]["dataset_directory"], 
                         config["data"]["train_stats"])
        )
    else:
        raise NotImplementedError()

    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        print("Loading weight file from {}".format(weight_file))
        network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )
    network.to(device)
    return network, train_on_batch, validate_on_batch


# set up learning scheduler
class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        raise NotImplementedError()


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, specs):
        print(specs)
        self.initial = specs["initial"]
        self.interval = specs["interval"]
        self.factor = specs["factor"]

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class LambdaLearningRateSchedule(LearningRateSchedule):
    def __init__(self, specs):
        print(specs)
        self.start_epoch = specs["start_epoch"]
        self.end_epoch = specs["end_epoch"]
        self.start_lr = specs["start_lr"]
        self.end_lr = specs["end_lr"]

    def lr_func(self, epoch):
        if epoch <= self.start_epoch:
            return 1.0
        elif epoch <= self.end_epoch:
            total = self.end_epoch - self.start_epoch
            delta = epoch - self.start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (self.end_lr / self.start_lr)
        else:
            return self.end_lr / self.start_lr

    def get_learning_rate(self, epoch):
        lambda_factor = self.lr_func(epoch)
        return self.start_lr * lambda_factor


class WarmupCosineLearningRateSchedule(LearningRateSchedule):
    def __init__(self, specs):
        print(specs)
        self.warmup_epochs = specs["warmup_epochs"]
        self.total_epochs = specs["total_epochs"]
        self.lr = specs["lr"]
        self.min_lr = specs["min_lr"]

    def get_learning_rate(self, epoch):
        if epoch <= self.warmup_epochs:
            lr = self.lr
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                (1.0 + math.cos(math.pi * (epoch-self.warmup_epochs) / \
                                (self.total_epochs - self.warmup_epochs)))
        return lr


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    if (type(lr_schedules)==list):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
    else:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules.get_learning_rate(epoch)


def schedule_factory(config):
    """Based on the provided config create the suitable learning schedule."""
    schedule = config.get("schedule", "lambda")

    # Set up LearningRateSchedule
    if schedule == "step" or schedule == "Step":
        lr_schedule = StepLearningRateSchedule({
                "type": "step",
                "initial" : config.get("lr", 1e-3),
                "interval": config.get("lr_step", 100),
                "factor"  : config.get("lr_decay", 0.1),
            },)

    elif schedule == "lambda" or schedule == "Lambda":
        lr_schedule = LambdaLearningRateSchedule({
                "type": "lambda",
                "start_epoch": config.get("start_epoch", 1000),
                "end_epoch"  : config.get("end_epoch", 1000),
                "start_lr"   : config.get("start_lr", 0.002),
                "end_lr"     : config.get("end_lr", 0.002),
            },)

    elif schedule == "warmupcosine" or schedule == "WarmupCosine":
        lr_schedule = WarmupCosineLearningRateSchedule({
                "type": "warmupcosine",
                "warmup_epochs" : config.get("warmup_epochs", 10),
                "total_epochs"  : config.get("total_epochs", 2000),
                "lr"            : config.get("lr", 2e-4),
                "min_lr"        : config.get("min_lr", 1e-6),
            },)

    else:
        raise NotImplementedError()

    return lr_schedule