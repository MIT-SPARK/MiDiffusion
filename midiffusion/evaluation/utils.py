import numpy as np
import torch
from torch.utils.data import dataloader
from tqdm import tqdm

from midiffusion.networks.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from midiffusion.networks.diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed
from midiffusion.datasets.threed_front_encoding import Diffusion


def get_feature_mask(network, experiment, num_known_objects, device):
    if experiment == "synthesis":
        feature_mask = None
        print("Experiment: scene synthesis.")
    elif experiment == "scene_completion":
        assert num_known_objects > 0
        feature_mask = torch.zeros(
            (network.sample_num_points, network.point_dim + network.class_dim),
            dtype=torch.bool, device=device
        )
        feature_mask[:num_known_objects] = True
        print("Experiment: scene completion (given {} objects) using corruption-and-masking."\
              .format(num_known_objects))
    elif experiment == "furniture_arrangement":
        feature_mask = torch.zeros(
            network.point_dim + network.class_dim, dtype=torch.bool, device=device
        )
        feature_mask[network.translation_dim: 
                     network.translation_dim + network.size_dim] = True # size
        feature_mask[network.bbox_dim: 
                     network.bbox_dim + network.class_dim] = True       # class
        feature_mask = feature_mask.repeat(network.sample_num_points, 1)
        print("Experiment: furniture arrangement.")
    elif experiment == "object_conditioned":
        feature_mask = torch.zeros(
            network.sample_num_points, network.point_dim + network.class_dim, 
            dtype=torch.bool, device=device
        )
        feature_mask[:, network.bbox_dim: network.bbox_dim + network.class_dim] = True       # class
        print("Experiment: object conditioned synthesis using corruption-and-masking.")
    elif experiment == "scene_completion_conditioned":
        feature_mask = torch.zeros(
            network.sample_num_points, network.point_dim + network.class_dim, 
            dtype=torch.bool, device=device
        )
        feature_mask[:num_known_objects] = True     # existing objects
        feature_mask[:, network.bbox_dim: network.bbox_dim + network.class_dim] = True       # class
        print("Experiment: scene completion (given {} objects) conditioned on labels using corruption-and-masking."\
              .format(num_known_objects))
    else:
        raise NotImplemented
    return feature_mask


def generate_layouts(network:DiffusionSceneLayout_DDPM, encoded_dataset:Diffusion, 
                     config, num_syn_scenes, sampling_rule="random", 
                     experiment="synthesis", num_known_objects=0, 
                     batch_size=16, device="cpu"):
    """Generate speicifed number of object layouts and also return a list of scene 
    indices corresponding to the floor plan. Each layout is a 2D array where each 
    row contain the concatenated object attributes.
    (Note: this code assumes "end" is the last object label, and, if used, 
    "start" is the second to last label.)"""
    
    # Sample floor layout
    if sampling_rule == "random":
        sampled_indices = np.random.choice(len(encoded_dataset), num_syn_scenes).tolist()
    elif sampling_rule == "uniform":
        sampled_indices = np.arange(len(encoded_dataset)).tolist() * \
            (num_syn_scenes // len(encoded_dataset))
        sampled_indices += \
            np.random.choice(len(encoded_dataset), 
                             num_syn_scenes - len(sampled_indices)).tolist()
    else:
        raise NotImplemented
    
    # network params
    with_room_mask = config["network"].get("room_mask_condition", True)
    print("Floor condition: {}.".format(with_room_mask))
    feature_mask = get_feature_mask(network, experiment, num_known_objects, device)
    
    # Generate layouts
    network.to(device)
    network.eval()
    layout_list = []
    for i in tqdm(range(0, num_syn_scenes, batch_size)):
        scene_indices = sampled_indices[i: min(i + batch_size, num_syn_scenes)]
        
        room_feature = None
        if with_room_mask:
            if config["feature_extractor"]["name"] == "resnet18":
                room_feature = torch.from_numpy(np.stack([
                    encoded_dataset[ind]["room_layout"] for ind in scene_indices
                ], axis=0)).to(device)
            elif config["feature_extractor"]["name"] == "pointnet_simple":
                room_feature = torch.from_numpy(np.stack([
                    encoded_dataset[ind]["fpbpn"] for ind in scene_indices
                ], axis=0)).to(device)
                
        if experiment == "synthesis":
            input_boxes = None
        else:
            samples = list(encoded_dataset[ind] for ind in scene_indices)
            sample_params = dataloader.default_collate(samples)
            input_boxes = network.unpack_data(sample_params).to(device)

        bbox_params_list = network.generate_layout(
            room_feature=room_feature,
            batch_size=len(scene_indices),
            input_boxes=input_boxes,
            feature_mask=feature_mask,
            device=device,
        )
        for bbox_params_dict in bbox_params_list:
            boxes = encoded_dataset.post_process(bbox_params_dict)
            bbox_params = {k: v.numpy()[0] for k, v in boxes.items()}
            layout_list.append(bbox_params)
    
    return sampled_indices, layout_list
