import torch

from .diffusion_mixed import MixedDiffusionPoint
from .denoising_net.mixed_transformer import MixedDenoiseTransformer
from .feature_extractors import ResNet18, PointNet_Point
from .diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM


class DiffusionSceneLayout_Mixed(DiffusionSceneLayout_DDPM):
    """Scene synthesis via mixed discrete-continuous diffusion"""
    def __init__(self, n_object_types, feature_extractor, config, train_stats_file):
        super(DiffusionSceneLayout_Mixed, self).__init__(
            n_object_types, feature_extractor, config, train_stats_file
        )
        
        # define the denoising network
        assert config["net_type"] == "transformer"
        denoise_net = MixedDenoiseTransformer(
            network_dim = self.network_dim,
            context_dim = self.context_dim,
            num_timesteps = config["time_num"],
            **config["net_kwargs"]
        )

        # define the diffusion type
        config["diffusion_geometric_kwargs"]["train_stats_file"] = train_stats_file
        self.diffusion = MixedDiffusionPoint(
            denoise_net = denoise_net,
            network_dim = self.network_dim,
            time_num = config["time_num"],
            d3pm_config = config["diffusion_semantic_kwargs"],
            ddpm_config = config["diffusion_geometric_kwargs"],
        )
        self.point_dim = sum(dim for k, dim in self.network_dim.items() 
                             if k !="class_dim")

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
        semantic_target = \
            room_layout_target[:, :, self.bbox_dim:self.bbox_dim+self.class_dim]\
                .argmax(dim=-1)
        geometric_target = torch.cat([
            room_layout_target[:, :, :self.bbox_dim], 
            room_layout_target[:, :, self.bbox_dim+self.class_dim:]
        ], dim=-1).contiguous()
        
        # denoise loss function
        loss, loss_dict = self.diffusion.get_loss_iter(
            semantic_target, geometric_target, condition=condition
        )
        return loss, loss_dict

    def sample(self, room_feature=None, batch_size=1, input_boxes=None, 
               feature_mask=None, clip_denoised=False, ret_traj=False, freq=40, 
               device="cpu"):
        # condition to denoise_net
        condition = self.unpack_condition(batch_size, device, room_feature)
        
        # retrieve known features from input_boxes if available
        x0_class, x0_geometric, class_mask, geometry_mask = None, None, None, None
        if input_boxes is not None:
            assert input_boxes.shape == torch.Size(
                (batch_size, self.sample_num_points, self.point_dim+self.class_dim)
            )
            assert feature_mask.shape == torch.Size(
                (self.sample_num_points, self.point_dim+self.class_dim)
            )
            class_mask = torch.all(
                feature_mask[:, self.bbox_dim:self.bbox_dim+self.class_dim],
                dim=1
            )           # 1-D
            geometry_mask = torch.cat([
                feature_mask[:, :self.bbox_dim],
                feature_mask[:, self.bbox_dim+self.class_dim:]
            ], dim=1)          # 2-D
                        
            if class_mask.any():
                x0_class = \
                    input_boxes[:, :, self.bbox_dim:self.bbox_dim+self.class_dim]\
                        .argmax(dim=-1)
            if geometry_mask.any():
                x0_geometric = torch.cat([
                    input_boxes[:, :, :self.bbox_dim],
                    input_boxes[:, :, self.bbox_dim+self.class_dim:]
                ], dim=-1)            

        # reverse sampling
        geometric_data_shape = (batch_size, self.sample_num_points, self.point_dim)
        if ret_traj:
            samples_traj = self.diffusion.gen_samples(
                geometric_data_shape, device=device, condition=condition,
                x0_class_partial=x0_class, class_mask=class_mask,
                x0_geometric_partial=x0_geometric, geometry_mask=geometry_mask, 
                freq=freq, clip_denoised=clip_denoised
            )
            samples_list = []
            for samples_class, samples_geometric in samples_traj:
                samples_list.append(torch.cat([
                    samples_geometric[:, :, :self.bbox_dim],
                    torch.exp(samples_class),
                    samples_geometric[:, :, self.bbox_dim:]
                ], dim=-1))
            return samples_list
        else:
            samples_class, samples_geometric = self.diffusion.gen_samples(
                geometric_data_shape, device=device, condition=condition,
                x0_class_partial=x0_class, class_mask=class_mask,
                x0_geometric_partial=x0_geometric, geometry_mask=geometry_mask, 
                clip_denoised=clip_denoised
            )
            return torch.cat([
                samples_geometric[:, :, :self.bbox_dim],
                torch.exp(samples_class),
                samples_geometric[:, :, self.bbox_dim:]
            ], dim=-1)
