#
# Modified from:
#   https://github.com/tangjiapeng/DiffuScene/blob/master/scene_synthesis/networks/diffusion_scene_layout_ddpm.py
# This version (1) simplifies some parts of the original implementation, and 
# (2) works with transformer denoising network as well as U-Net.
#

import torch
import torch.nn as nn
from torch.nn import Module

from .diffusion_ddpm import DiffusionPoint
from .denoising_net.unet1D import Unet1D
from .denoising_net.continuous_transformer import ContinuousDenoiseTransformer
from .feature_extractors import ResNet18, PointNet_Point


class DiffusionSceneLayout_DDPM(Module):
    """Scene synthesis via continuous state diffusion"""
    def __init__(self, n_object_types, feature_extractor, config, train_stats_file):
        super().__init__()
        self.n_object_types = n_object_types
        self.config = config

        if "class_dim" in config:
            assert config["class_dim"] == n_object_types + 1
        else:
            config["class_dim"] = n_object_types + 1
        
        # read object property dimension
        self.class_dim = config.get("class_dim")
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 1)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objectness_dim = config.get("objectness_dim", 0)
        self.objfeat_dim = config.get("objfeat_dim", 0)
        self.network_dim = {k: getattr(self, k) for k in [
            "objectness_dim", "class_dim", "translation_dim", "size_dim", 
            "angle_dim", "objfeat_dim"
        ]}

        # maximum number of points
        self.sample_num_points = config.get("sample_num_points", 12)

        # initialize conditinoal input dimension
        self.context_dim = 0

        # room_mask_condition: if yes, define the feature extractor for the room mask
        self.room_mask_condition = config.get("room_mask_condition", True)
        if self.room_mask_condition:
            self.room_latent_dim = config["room_latent_dim"]
            self.feature_extractor = feature_extractor
            self.fc_room_f = nn.Linear(
                self.feature_extractor.feature_size, self.room_latent_dim
            ) if self.feature_extractor.feature_size != self.room_latent_dim \
                else nn.Identity()
            print('use room mask as condition')
            self.context_dim += self.room_latent_dim
        
        # define positional embeddings
        self.position_condition = config.get("position_condition", False)
        self.learnable_embedding = config.get("learnable_embedding", False)
        
        if self.position_condition:
            self.position_emb_dim = config.get("position_emb_dim", 64)
            if self.learnable_embedding:
                self.register_parameter("positional_embedding", nn.Parameter(
                        torch.randn(self.sample_num_points, self.position_emb_dim)
                    ))
            else:
                self.fc_position_condition = nn.Sequential(
                    nn.Linear(self.sample_num_points, self.position_emb_dim, 
                              bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.position_emb_dim, self.position_emb_dim, 
                              bias=False),
                )
            print('use position embedding for object index')
            self.context_dim += self.position_emb_dim
        
        if "diffusion_kwargs" in config.keys():
            # define the denoising network
            if config["net_type"] == "unet1d":
                denoise_net = Unet1D(
                    network_dim = self.network_dim,
                    context_dim = self.context_dim,
                    **config["net_kwargs"]
                )
            elif config["net_type"] == "transformer":
                denoise_net = ContinuousDenoiseTransformer(
                    network_dim = self.network_dim,
                    context_dim = self.context_dim,
                    num_timesteps = config["diffusion_kwargs"]["time_num"],
                    **config["net_kwargs"]
                )
            else:
                raise NotImplementedError()

            # define the diffusion type
            self.diffusion = DiffusionPoint(
                denoise_net = denoise_net,
                network_dim = self.network_dim,
                train_stats_file = train_stats_file,
                **config["diffusion_kwargs"]
            )
        self.point_dim = sum(dim for dim in self.network_dim.values())
    
    def unpack_data(self, sample_params):
        # read data
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        if self.objectness_dim > 0:
            objectness = sample_params["objectness"]
        if self.objfeat_dim == 32:
            objfeats = sample_params["objfeats_32"]
        elif self.objfeat_dim == 64:
            objfeats = sample_params["objfeats"]
        elif self.objfeat_dim != 0:
            raise NotImplemented

        # get desired diffusion target
        room_layout_target = \
            torch.cat([translations, sizes, angles, class_labels], dim=-1)
        if self.objectness_dim > 0:
            room_layout_target = \
                torch.cat([room_layout_target, objectness], dim=-1)
        if self.objfeat_dim > 0:
            room_layout_target = \
                torch.cat([room_layout_target, objfeats], dim=-1)
        
        return room_layout_target
    
    def unpack_condition(self, batch_size, device, room_feature=None):        
        # condition to denoise_net
        condition = None

        # get the latent feature of room_mask
        if self.room_mask_condition:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_feature))
            condition = room_layout_f[:, None, :].repeat(1, self.sample_num_points, 1)
        
        # process instance position condition f
        if self.position_condition:
            if self.learnable_embedding:
                position_condition_f = self.positional_embedding[None, :]\
                    .repeat(batch_size, 1, 1)
            else:
                instance_label = torch.eye(self.sample_num_points).float()\
                    .to(device)[None, ...].repeat(batch_size, 1, 1)
                position_condition_f = self.fc_position_condition(instance_label) 
            
            condition = torch.cat(
                [condition, position_condition_f], dim=-1
            ).contiguous() if condition is not None else position_condition_f
        
        return condition

    def get_loss(self, sample_params):
        # unpack sample_params
        room_layout_target = self.unpack_data(sample_params)
        
        # unpack condition
        room_feature = None
        if self.room_mask_condition:
            if isinstance(self.feature_extractor, ResNet18):
                room_feature = sample_params["room_layout"]
            elif isinstance(self.feature_extractor, PointNet_Point):
                room_feature = sample_params["fpbpn"]
        condition = self.unpack_condition(
            room_layout_target.shape[0], room_layout_target.device, room_feature
        )
        
        # denoise loss function
        loss, loss_dict = self.diffusion.get_loss_iter(
            room_layout_target, condition=condition
        )
        return loss, loss_dict

    def sample(self, room_feature=None, batch_size=1, input_boxes=None, 
               feature_mask=None, clip_denoised=False, ret_traj=False, freq=40, 
               device="cpu"):
        # condition to denoise_net
        condition = self.unpack_condition(batch_size, device, room_feature)

        # reverse sampling
        data_shape = (batch_size, self.sample_num_points, self.point_dim)          
        return self.diffusion.gen_samples(
            data_shape, device=device, condition=condition,
            clip_denoised=clip_denoised, freq=freq if ret_traj else None
        )

    @torch.no_grad()
    def generate_layout(self, room_feature=None, batch_size=1, input_boxes=None,
                        feature_mask=None, clip_denoised=False, device="cpu"):
        """Generate a list of bbox_params dict, each corresponds to one layout 
        that can be processed by dataset's post_process() class function.
        The features in each dict is a tensor of [0, Ni, ?] dimension where 
        Ni is the number of objects predicted."""
        if self.room_mask_condition:
            assert room_feature.size(0) == batch_size
        
        samples = self.sample(
            room_feature, batch_size, input_boxes=input_boxes, feature_mask=feature_mask,
            clip_denoised=clip_denoised, device=device, ret_traj=False
        )
        
        return self.delete_empty_from_network_samples(samples)

    @torch.no_grad()
    def generate_layout_progressive(self, room_feature=None, batch_size=1, 
                                    input_boxes=None, feature_mask=None, 
                                    clip_denoised=False, device="cpu", 
                                    save_freq=100):
        """Generate a list of tuples. Each tuple stores a trajectory of one 
        predicted layout. It contains a collection of bbox_params dict, each 
        corresdpons to a frame in the reverse diffusion process that can be 
        processed by dataset's post_process() class function.
        The features in each dict is a tensor of [0, Ni, ?] dimension where 
        Ni is the number of objects predicted."""
        # generate results at each time step
        samples_traj = self.sample(
            room_feature, batch_size, input_boxes=input_boxes, feature_mask=feature_mask,
            clip_denoised=clip_denoised, device=device, ret_traj=True, freq=save_freq,  
        )
        results_by_time = []
        for samples_t in samples_traj:
            samples_t_list = self.delete_empty_from_network_samples(samples_t)
            results_by_time.append(samples_t_list)

        # combine results of the same frame to a tuple
        traj_list = []
        for b in range(batch_size):
            traj_list.append(tuple(results[b] for results in results_by_time))
        
        return traj_list

    @torch.no_grad()
    def delete_empty_from_network_samples(self, samples):
        """Remove objects with 'empty' label given samples of [B, N, C] 
        dimensions. The output is a list of dictionaries with features 
        of [0, N_i, ?] dimensions for each object feature."""
        object_max, object_max_ind = torch.max(
            samples[:, :, self.bbox_dim:self.bbox_dim+self.n_object_types], 
            dim=-1
        )
        samples_dict = {
            "translations": samples[:, :, 0:self.translation_dim].contiguous(),
            "sizes": samples[:, :,  self.translation_dim:self.translation_dim+
                             self.size_dim].contiguous(),
            "angles": samples[:, :, self.translation_dim+self.size_dim:
                              self.bbox_dim].contiguous(),
            "class_labels": nn.functional.one_hot(
                object_max_ind, num_classes=self.n_object_types
            ),
            "is_empty": samples[:, :, self.bbox_dim+self.class_dim-1] > object_max,
        }
        if self.objfeat_dim > 0:
            samples_dict["objfeats"] = \
                samples[:, :, self.bbox_dim+self.class_dim:self.bbox_dim+
                        self.class_dim+self.objfeat_dim]

        return self.delete_empty_boxes(samples_dict)

    @torch.no_grad()
    def delete_empty_boxes(self, samples_dict):
        """Remove objects with 'empty' label given samples_dict with features of 
        [B, N, ?] dimensions. The output is a list of dictionaries with features 
        of [0, N_i, ?] dimensions for each object feature."""
        batch_size = samples_dict["class_labels"].size(0)
        max_boxes = samples_dict["class_labels"].size(1)

        return_properties = ["class_labels", "translations", "sizes", "angles"]
        if self.objfeat_dim > 0:
            return_properties.append("objfeats")
        
        boxes_list = []
        for b in range(batch_size):
            # Initialize empty dict
            boxes = {
                "class_labels": torch.zeros(1, 0, self.n_object_types),
                "translations": torch.zeros(1, 0, self.translation_dim),
                "sizes": torch.zeros(1, 0, self.size_dim),
                "angles": torch.zeros(1, 0, self.angle_dim)
            }
            if self.objfeat_dim > 0:
                boxes["objfeats"] = torch.zeros(1, 0, self.objfeat_dim)
            
            for i in range(max_boxes):
                # Check if we have the end symbol
                if samples_dict["is_empty"][b, i]:
                    continue
                else:
                    for k in return_properties:
                        boxes[k] = torch.cat([
                            boxes[k], 
                            samples_dict[k][b:b+1, i:i+1, :].to("cpu")
                        ], dim=1)
            
            boxes_list.append(boxes)

        return boxes_list
