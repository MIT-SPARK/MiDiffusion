import inspect
import torch.nn as nn
import torch.utils.data
from .diffusion_base import BaseDiffusion, LOG_ZERO
from .diffusion_d3pm import MaskAndReplaceDiffusion, alpha_schedule, index_to_log_onehot, log_onehot_to_index
from .diffusion_ddpm import GaussianDiffusion, get_betas


def extract_params(func, param_dict):
    func_args = inspect.signature(func).parameters
    return {k:v for k, v in param_dict.items() if k in func_args}


class MixedDiffusionPoint(nn.Module):
    def __init__(self, denoise_net:nn.Module, network_dim, time_num,
                 d3pm_config, ddpm_config):          
        super(MixedDiffusionPoint, self).__init__()

        self.num_timesteps = time_num
        self.num_classes = network_dim["class_dim"]     # object categories plus [empty]
        
        # discrete semantic diffusion
        noise_params = alpha_schedule(
            num_timesteps=time_num, N=network_dim["class_dim"], 
            **extract_params(alpha_schedule, d3pm_config)
        )
        self.diffusion_semantic = MaskAndReplaceDiffusion(
            network_dim["class_dim"] + 1, noise_params, 
            **extract_params(MaskAndReplaceDiffusion.__init__, d3pm_config)
        )
        assert self.diffusion_semantic.num_classes == self.num_classes + 1
        assert self.diffusion_semantic.model_output_type == "x0"

        # continuous geometric diffusion
        network_dim["class_dim"] = 0
        betas = get_betas(
            time_num=time_num, 
            **extract_params(get_betas, ddpm_config)
        )
        self.diffusion_geometric = GaussianDiffusion(
            network_dim, betas,
            **extract_params(GaussianDiffusion.__init__, ddpm_config)
        )
        assert self.diffusion_geometric.loss_type == "mse"

        # denoising net that takes in semantic and geometric features, 
        # and output corresponding discrete and continuout predictions
        self.model = denoise_net

    def _denoise(self, data_semantic, data_geometric, t, condition, out_type="all"):
        out_class, out_bbox = \
            self.model(data_semantic, data_geometric, t, condition)
        if out_type == "semantic":
            return out_class
        elif out_type == "geometric":
            return out_bbox
        else:
            return out_class, out_bbox

    def get_loss_iter(self, data_semantic, data_geometric, condition=None):
        B, N, C = data_geometric.shape
        device = data_geometric.device
        assert data_semantic.shape == (B, N)
        assert data_semantic.device == device
        
        # Move models and pre-computed tensors to data device
        self.diffusion_semantic._move_tensors(device)
        self.diffusion_geometric._move_tensors(device)
        self.model.to(device)

        # Sample q(x_t | x_0)
        t = torch.randint(0, self.num_timesteps, size=(B,), device=device)
        # x_t_class: (B, N)
        log_xstart = index_to_log_onehot(data_semantic, self.num_classes + 1)
        log_xt = self.diffusion_semantic.q_sample(log_x_start=log_xstart, t=t)
        x_t_class = log_onehot_to_index(log_xt)
        # x_t_geometric: (B, N, C)
        noise_geometric = torch.randn_like(data_geometric)
        x_t_geometric = self.diffusion_geometric.q_sample(
            x_0=data_geometric, t=t, eps=noise_geometric
        )

        # Send x_t, t, condition through denoising net
        denoise_out_class, denoise_out_geometric = \
            self.model(x_t_class, x_t_geometric, t, context=condition)
        
        # Compute loss
        feat_separated_losses = dict()  # loss.<feat_name>: <loss_per_scene>

        # semantic
        log_x0_recon = self.diffusion_semantic.log_pred_from_denoise_out(denoise_out_class) # p_theta(x0|xt)
        log_p_prob = self.diffusion_semantic.q_posterior(
                log_x_start=log_x0_recon, log_x_t=log_xt, t=t
            )   # go through q(xt_1| xt, x0)
        # semantic - train loss
        loss_tensor = self.diffusion_semantic.compute_kl_loss(log_xstart, log_xt, t, log_p_prob)
        if self.diffusion_semantic.mask_weight != 1:   # adjust [mask] token weight
            mask_token_region = (x_t_class == self.num_classes)
            loss_tensor = torch.where(
                mask_token_region, 
                self.diffusion_semantic.mask_weight * loss_tensor, 
                loss_tensor
            )
        loss_class = loss_tensor.mean(dim=1)    # average over (B, N) loss tensor
        feat_separated_losses["loss.class"] = loss_class
        # semantic - aux loss
        if self.diffusion_semantic.auxiliary_loss_weight > 0:
            loss_tensor = self.diffusion_semantic.compute_aux_loss(log_xstart, log_x0_recon, t)
            if self.diffusion_semantic.mask_weight != 1:   # adjust [mask] token weight
                loss_tensor = torch.where(
                    mask_token_region, 
                    self.diffusion_semantic.mask_weight * loss_tensor, 
                    loss_tensor
                )
            aux_loss = loss_tensor.mean(dim=1)  # average over (B, N) loss tensor
            if self.diffusion_semantic.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0
            loss_class_aux = (addition_loss_weight * \
                self.diffusion_semantic.auxiliary_loss_weight * aux_loss).mean(dim=-1)
        else:
            loss_class_aux = torch.zeros(B, device=device)
        feat_separated_losses["loss.class_aux"] = loss_class_aux

        # geometric
        object_mask = None  # TODO
        # object_mask = torch.logical_and(
        #     (x_t_class != self.num_classes), (x_t_class != self.num_classes - 1)
        # )
        loss_tensor = self.diffusion_geometric.compute_mse_loss(
            data_geometric, x_t_geometric, t, denoise_out_geometric, noise_geometric
        )
        weight_at_t = BaseDiffusion._extract(
            self.diffusion_geometric.loss_weight, t, torch.Size([B])
        )
        for feat_name, (feat_dim, start_ind) in self.diffusion_geometric.dim_dict.items():
            if feat_name in ["class", "object"] or feat_dim == 0:
                continue
            feat_loss = BaseDiffusion.mean_per_batch(
                loss_tensor, feat_dim, start_ind, mask=object_mask
            )
            feat_separated_losses["loss." + feat_name] = feat_loss * weight_at_t
        loss_geometric = feat_separated_losses["loss.bbox"]
        if "loss_objfeat" in feat_separated_losses.keys():
            loss_geometric += feat_separated_losses["loss.objfeat"]
        # additional bounding box regularization loss on mean of p(x_{t-1} | x_t)
        if self.diffusion_geometric.loss_iou:
            x_recon = self.diffusion_geometric.predict_start_from_denoise_out(
                denoise_out_geometric, x_t_geometric, t, clip_x_start=False
            )
            loss_iou = self.diffusion_geometric.bbox_iou_losses(x_recon, t)
            feat_separated_losses['loss.liou'] = loss_iou
        else:
            loss_iou = 0
            
        return ((loss_class + loss_class_aux + loss_geometric + loss_iou).mean(),
                {k: v.mean() for k, v in feat_separated_losses.items()})
    
    def gen_samples(self, shape_geo, device, condition, 
                    x0_class_partial=None, class_mask=None,
                    x0_geometric_partial=None, geometry_mask=None,
                    freq=None, clip_denoised=False):
        B, N, C = shape_geo
        self.diffusion_semantic._move_tensors(device)
        self.diffusion_geometric._move_tensors(device)
        self.model.to(device)
        
        # Generate complete diffusion of input partial data
        if x0_class_partial is None:
            partial_class = False
        else:
            partial_class = True
            assert x0_class_partial.shape == torch.Size((B, N))
            assert len(class_mask) == N

            # create a full sequence of noisy x_t's
            log_x_t_class_partials = []
            log_x_t_class_partial = index_to_log_onehot(x0_class_partial, self.num_classes + 1)
            for t in range(self.num_timesteps):
                t_ = torch.full((B,), t, dtype=torch.int64, device=device)
                log_x_t_class_partial = self.diffusion_semantic\
                    .q_sample_one_step(log_x_t_class_partial, t_, no_mask=True)
                log_x_t_class_partials.append(log_x_t_class_partial)

        if x0_geometric_partial is None:
            partial_geometry = False
        else:
            partial_geometry = True
            assert x0_geometric_partial.shape == torch.Size((B, N, C))
            assert geometry_mask.shape == torch.Size((N, C))

            # create a full sequence of noisy x_t's
            x_t_geometric_partials = []
            x_t_geometric_partial = x0_geometric_partial
            for t in range(self.num_timesteps):
                t_ = torch.full((B,), t, dtype=torch.int64, device=device)
                x_t_geometric_partial = self.diffusion_geometric\
                    .q_sample_one_step(x_t_geometric_partial, t_)
                x_t_geometric_partials.append(x_t_geometric_partial)
        
        # Initialize data
        log_zeros = torch.full((B, self.num_classes, N), LOG_ZERO, device=device)
        log_ones = torch.zeros((B, 1, N), device=device)
        log_x_end_class = torch.cat((log_zeros, log_ones), dim=1)
        x_end_geometric = torch.randn(size=shape_geo, dtype=torch.float, device=device)
        
        # Backward denoising steps
        if freq is not None:
            pred_traj = []
        log_x_t_class = log_x_end_class     # (B, num_classes+1, N) - flip last two axis for output
        x_t_geometric = x_end_geometric     # (B, N, num_features)
        for t in reversed(range(0, self.num_timesteps)):
            t_ = torch.full((B,), t, dtype=torch.int64, device=device)

            if partial_class:
                log_x_t_class[:, :, class_mask] = log_x_t_class_partials[t][:, :, class_mask]
            if partial_geometry:
                x_t_geometric = torch.where(
                    geometry_mask.unsqueeze(0).expand_as(x_t_geometric),
                    x_t_geometric_partials[t],
                    x_t_geometric
                )
            
            x_t_class = log_onehot_to_index(log_x_t_class)
            denoise_out_class, denoise_out_geometric = \
                self.model(x_t_class, x_t_geometric, t_, context=condition)
            # semantic label probability distribution
            log_x_recon = self.diffusion_semantic.log_pred_from_denoise_out(denoise_out_class)
            if t == 0:
                log_EV_qxt_x0 = log_x_recon
            else:
                log_EV_qxt_x0 = self.diffusion_semantic.q_posterior(
                    log_x_start=log_x_recon, log_x_t=log_x_t_class, t=t_
                )
            # semantic label sampling
            log_x_t_class = self.diffusion_semantic.log_sample_categorical(log_EV_qxt_x0)
            # geometric feature probability distribution
            p_mean, _, p_log_var = self.diffusion_geometric\
                .p_pred_from_denoise_out(denoise_out_geometric, x_t_geometric, t_)
            # geometric feature sampling
            noise = torch.randn_like(p_mean)
            x_t_geometric = p_mean + torch.exp(0.5 * p_log_var) * noise
            if clip_denoised:
                x_t_geometric = self.diffusion_geometric._clip_bbox(x_t_geometric)
        
            if freq is not None and (t % freq == 0 or t == self.num_timesteps - 1):
                pred_traj.append(
                    (log_x_t_class[:, :-1, :].permute(0, 2, 1), x_t_geometric)
                )   # (B, N, num_classes), (B, N, num_features)
        
        # Replace final prediction with partial inputs
        if partial_class:
            log_x_t_class[:, :, class_mask] = \
                index_to_log_onehot(x0_class_partial, self.num_classes + 1)[:, :, class_mask]
        if partial_geometry:
            x_t_geometric = torch.where(
                geometry_mask.unsqueeze(0).expand_as(x_t_geometric),
                x0_geometric_partial,
                x_t_geometric
            )

        if (partial_class or partial_geometry) and freq is not None:
            pred_traj[-1] = (
                (log_x_t_class[:, :-1, :].permute(0, 2, 1), x_t_geometric)
            )
        
        if freq is not None:
            return pred_traj
        else:
            return log_x_t_class[:, :-1, :].permute(0, 2, 1), x_t_geometric
