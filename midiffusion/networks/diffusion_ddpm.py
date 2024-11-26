#
# Modified from:
#   https://github.com/tangjiapeng/DiffuScene
#

import torch.nn as nn
import torch.utils.data
import math
import json
from .loss import axis_aligned_bbox_overlaps_3d
from .diffusion_base import BaseDiffusion, LOG_ZERO

'''
helper functions
'''

def get_betas(schedule_type, beta_start, beta_end, time_num, warm_ratio=0.1):
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, time_num)
    elif schedule_type == 'warm':
        betas = beta_end * torch.ones(time_num)
        warmup_time = int(time_num * warm_ratio)
        betas[:warmup_time] = \
            torch.linspace(beta_start, beta_end, warmup_time)
    elif schedule_type == 'sigmoid':
        betas = torch.sigmoid(torch.linspace(-6, 6, time_num)) * \
            (beta_end - beta_start) + beta_start
    elif schedule_type == 'cosine':
        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            """
            Create a beta schedule that discretizes the given alpha_t_bar function,
            which defines the cumulative product of (1-beta) over time from t = [0,1].
            :param num_diffusion_timesteps: the number of betas to produce.
            :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                            produces the cumulative product of (1-beta) up to that
                            part of the diffusion process.
            :param max_beta: the maximum beta to use; use values lower than 1 to
                            prevent singularities.
            """
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            
            return torch.tensor(betas)
        
        betas_for_alpha_bar(
            time_num, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(schedule_type)
    return betas


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))


class GaussianDiffusion(BaseDiffusion):
    def __init__(self, network_dim, betas, loss_type="mse", model_mean_type="eps", 
                 model_var_type="fixedsmall", loss_separate=True, loss_iou=False,
                 train_stats_file=None):
        super().__init__()

        # read object property dimension
        self.objectness_dim, self.class_dim, self.objfeat_dim = \
            network_dim["objectness_dim"], network_dim["class_dim"], \
                network_dim["objfeat_dim"]
        self.translation_dim, self.size_dim, self.angle_dim = \
            network_dim["translation_dim"], network_dim["size_dim"], \
                network_dim["angle_dim"]
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.dim_dict = {
            "trans": (self.translation_dim, 0),
            "size": (self.size_dim, self.translation_dim),
            "angle": (self.angle_dim, self.translation_dim + self.size_dim),
            "bbox": (self.bbox_dim, 0),
            "class": (self.class_dim, self.bbox_dim),
            "object": (self.objectness_dim, self.bbox_dim + self.class_dim),
            "objfeat": (self.objfeat_dim, 
                        self.bbox_dim + self.class_dim + self.objectness_dim),
        }

        self.loss_separate = loss_separate
        self.loss_iou = loss_iou
        if self.loss_iou:
            with open(train_stats_file, "r") as f:
                train_stats = json.load(f)
            bounds_translations = train_stats["bounds_translations"]
            self._centroids_min = torch.tensor(bounds_translations[:3])
            self._centroids_max = torch.tensor(bounds_translations[3:])
            print('load centroids min {} and max {} in Gaussian Diffusion' \
                  .format(self._centroids_min, self._centroids_max))
            
            bounds_sizes = train_stats["bounds_sizes"]
            self._sizes_min = torch.tensor(bounds_sizes[:3])
            self._sizes_max = torch.tensor(bounds_sizes[3:])
            print('load sizes min {} and max {} in Gaussian Diffusion' \
                  .format(self._sizes_min, self._sizes_max))
            
            bounds_angles = train_stats["bounds_angles"]
            self._angles_min = bounds_angles[0]
            self._angles_max = bounds_angles[1]
            print('load angles min {} and max {} in Gaussian Diffusion' \
                  .format(self._angles_min, self._angles_max))

        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        # assert isinstance(betas, np.ndarray)
        # timesteps, = betas.shape
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = betas.shape[0]

        self.betas = betas.float()
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.]), self.alpha_bar[:-1]])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # mean = sqrt_alpha_t * x_{t-1}, variance = beta_t * Identity
        self.sqrt_recip_alpha = torch.sqrt(1. / self.alphas)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)
        self.log_one_minus_alpha_bar = torch.log(1. - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1. /self.alpha_bar)
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1. / self.alpha_bar - 1).float()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = \
            betas * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.clamp(
            torch.log(self.posterior_variance), min=LOG_ZERO
        )
        # mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        self.posterior_mean_coef1 = \
            betas * torch.sqrt(self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_mean_coef2 = \
            (1 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1 - self.alpha_bar)
        
        # calculate loss weight
        if model_mean_type == 'eps':    # noise of sampling from q(x_t | x_0)
            loss_weight = torch.ones_like(self.alpha_bar)
        elif model_mean_type == 'x0':   # x_0
            loss_weight = self.alpha_bar / (1 - self.alpha_bar)
        elif model_mean_type == 'v':    # x_0 = sqrt(alpha_bar_t)*x_t - sqrt(1- alpha_bar_t)*v_t
            loss_weight = self.alpha_bar
        elif model_mean_type == 'mu':   # mean of p_net(x_{t-1} | x_t)
            loss_weight = torch.ones_like(self.alpha_bar)
        else:
            raise NotImplemented
        self.loss_weight = loss_weight
    
    def _clip_bbox(self, data):
        # Clip bbox features to [-1, 1]
        return torch.cat((
            torch.clamp(data[:, :, :self.bbox_dim], -1, 1),
            data[:, :, self.bbox_dim:]
        ), dim=2)
    
    def _predict_start_from_eps(self, x_t, t, eps):
        # Assume parametrization:
        #   x_t(x_0, eps) = sqrt(alpha_bar_t)*x_0 + sqrt(1- alpha_bar_t)*eps
        # Predict x_0 given x_t, t, eps
        assert x_t.shape == eps.shape
        return self._extract(self.sqrt_recip_alpha_bar, t, x_t.shape) * x_t \
            - self._extract(self.sqrt_recipm1_alpha_bar, t, x_t.shape) * eps

    def _predict_start_from_v(self, x_t, t, v):
        #   x_0 = sqrt(alpha_bar_t)*x_t - sqrt(1- alpha_bar_t)*v_t
        return self._extract(self.sqrt_alpha_bar, t, x_t.shape) * x_t \
            - self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape) * v
    
    def _predict_start_from_mu(self, x_t, t, mu):
        #   mu_t = posterior_mean_coef1*x_0 + posterior_mean_coef2*x_t
        return (mu - self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t) / \
            self._extract(self.posterior_mean_coef1, t, x_t.shape)
    
    def _predict_v_from_start(self, x0, t, eps):
        #   v_t(x_0, eps) = sqrt(alpha_bar_t)*eps - sqrt(1- alpha_bar_t)*x_0
        return self._extract(self.sqrt_alpha_bar, t, x0.shape) * eps \
            - self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * x0
    
    def _predict_eps_from_v(self, x_t, t, v):
        #   eps = sqrt(1-alpha_bar_t)*x_t + sqrt(alpha_bat_t)*v_t
        return self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape) * x_t \
            + self._extract(self.sqrt_alpha_bar, t, v.shape) * v
    
    def q_pred_one_step(self, x_t_1, t):  
        """
        diffusion step: q(x_t | x_{t-1})
        """
        x_shape = x_t_1.shape
        mean = self._extract(torch.sqrt(self.alphas), t, x_shape) * x_t_1
        variance = self._extract(1. - self.alphas, t, x_shape)
        log_variance = torch.log(variance)
        return mean, variance, log_variance
    
    def q_pred(self, x_0, t):  
        """
        diffusion the data to time t: q(x_t | x_0)
        """
        x_shape = x_0.shape
        mean = self._extract(self.sqrt_alpha_bar, t, x_shape) * x_0
        variance = self._extract(1. - self.alpha_bar, t, x_shape)
        log_variance = self._extract(self.log_one_minus_alpha_bar, t, x_shape)
        return mean, variance, log_variance
    
    def q_sample_one_step(self, x_t_1, t, eps=None):
        """
        Sample from q(x_t | x_{t-1})
        """
        mean, _, log_variance = self.q_pred_one_step(x_t_1, t)
        if eps is None:
            eps = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_variance) * eps

    def q_sample(self, x_0, t, eps=None):
        """
        Sample from q(x_t | x_0)
        """
        mean, _, log_variance = self.q_pred(x_0, t)
        if eps is None:
            eps = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_variance) * eps
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean and variance of the posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        x_shape = x_0.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_shape) * x_0
                + self._extract(self.posterior_mean_coef2, t, x_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_shape
        )
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == \
                posterior_log_variance_clipped.shape[0] == x_0.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_denoise_out(self, denoise_out, x_t, t, clip_x_start=False):
        """
        Use output of denoising net to predict x_0;
        denoise_out = denoise_net(x_t, t, **kwargs)
        """
        if self.model_mean_type == 'eps':
            x_start = self._predict_start_from_eps(x_t, t, denoise_out)
        elif self.model_mean_type == 'x0':
            x_start = denoise_out
        elif self.model_mean_type == 'v':
            x_start = self._predict_start_from_v(x_t, t, denoise_out)
        elif self.model_mean_type == 'mu':
            x_start = self._predict_start_from_mu(x_t, t, denoise_out)
        else:
            raise NotImplemented

        if clip_x_start:
            return self._clip_bbox(x_start)
        else:
            return x_start
        
    def p_pred_from_denoise_out(self, denoise_out, x_t, t):
        """
        Convert output of denoising network to mean and variance
        """
        if self.model_mean_type == 'eps':
            p_mean = self.p_mean_from_eps(x_t, t=t, eps=denoise_out)
        elif self.model_mean_type == 'x0':
            p_mean, _, _ = self.q_posterior(x_0=denoise_out, x_t=x_t, t=t)
        elif self.model_mean_type == 'v':
            eps = self._predict_eps_from_v(x_t, t=t, v=denoise_out)
            p_mean = self.p_mean_from_eps(x_t, t=t, eps=eps)
        elif self.model_mean_type == 'mu':
            p_mean = denoise_out
        else:
            raise NotImplemented
        
        # Use fixed variance
        if self.model_var_type == "fixedlarge":
            p_variance = self.betas
            p_log_variance = torch.clamp(torch.log(p_variance), min=LOG_ZERO)
        elif self.model_var_type == "fixedsmall":
            p_variance = self.posterior_variance
            p_log_variance = self.posterior_log_variance_clipped
        else:
            raise NotImplemented
        p_variance = self._extract(p_variance, t, x_t.shape)\
            * torch.ones_like(x_t)
        p_log_variance = self._extract(p_log_variance, t, x_t.shape)\
            * torch.ones_like(x_t)

        return p_mean, p_variance, p_log_variance

    def p_pred(self, denoise_fn, x_t, t, condition, condition_cross):
        """
        mean and variance of the denoising step p(x_{t-1} | x_t)
        """
        # Compute prediceted mean
        model_output = denoise_fn(x_t, t, condition, condition_cross)
        p_mean, p_variance, p_log_variance = \
            self.p_pred_from_denoise_out(model_output, x_t, t)
        assert p_mean.shape == x_t.shape
        assert p_variance.shape == x_t.shape
        assert p_log_variance.shape == x_t.shape

        return p_mean, p_variance, p_log_variance

    def p_mean_from_eps(self, x_t, t, eps):
        """
        mean of p(x_{t-1} | x_t) given x_t and eps
        """
        eps_coeffs = self._extract(
            self.betas / self.sqrt_one_minus_alpha_bar, t, eps.shape
        )
        return self._extract(self.sqrt_recip_alpha, t, x_t.shape) * (
                x_t - eps_coeffs * eps
        )

    def p_sample(self, denoise_fn, x_t, t, condition, condition_cross, 
                 clip_denoised=False):
        """
        denoise step: p(x_{t-1} | x_t)
        """
        mean, _, log_variance = self.p_pred(
            denoise_fn, x_t=x_t, t=t, condition=condition, 
            condition_cross=condition_cross
        )

        noise = torch.randn_like(mean)
        sample = mean + torch.exp(0.5 * log_variance) * noise

        if clip_denoised:
            return self._clip_bbox(sample)
        else:
            return sample

    def p_sample_loop(self, denoise_fn, x_end, condition, condition_cross,
                      x_target_fn=lambda x: x, log_freq=None, clip_denoised=True):
        """
        Generate samples through iterative denoising.
        """
        B = x_end.shape[0]
        if log_freq:
            x_traj = [x_end]
        
        x_t = x_end
        total_steps = self.num_timesteps
        for t in reversed(range(0, total_steps)):
            t_ = torch.full((B,), t, dtype=torch.int64, device=self.device)
            x_t = self.p_sample(
                denoise_fn=denoise_fn, x_t=x_t, t=t_, condition=condition, 
                condition_cross=condition_cross, clip_denoised=clip_denoised
            )
            x_t = x_target_fn(x_t)

            if log_freq and t % log_freq == 0:
                x_traj.append(x_t)

        if log_freq:
            return x_traj
        else:
            return x_t

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, t, noise=None,
                      condition=None, condition_cross=None):
        """
        variational bound at x_0 and t
        """
        if noise is None:
            noise = torch.rand_like(data_start)
        x_t = self.q_sample(data_start, t, eps=noise)

        posterior_mean, _, posterior_log_variance_clipped = \
            self.q_posterior(x_0=data_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance = self.p_pred(
            denoise_fn, x_t=x_t, t=t, condition=condition, 
            condition_cross=condition_cross
        )
        kl = normal_kl(posterior_mean, posterior_log_variance_clipped, 
                       model_mean, model_log_variance)
        assert kl.shape == data_start.shape
        return kl

    def compute_mse_loss(self, x_0, x_t, t, denoise_out, noise=None):
        assert self.loss_type == 'mse'     # simplified loss
        
        if self.model_mean_type == 'eps':
            target = noise
        elif self.model_mean_type == 'x0':
            target = x_0
        elif self.model_mean_type == 'v':
            target = self._predict_v_from_start(x_0, t, noise)
        elif self.model_mean_type == 'mu':
            target = self.q_posterior(x_0, x_t, t)[0]
        else:
            raise NotImplemented
        
        return (target - denoise_out) ** 2

    def p_losses(self, denoise_fn, data_start, t=None, noise=None, 
                 condition=None, condition_cross=None):
        """
        Training loss calculation
        """
        B = data_start.shape[0]
        assert data_start.shape[-1] == \
            self.objectness_dim+self.class_dim+self.bbox_dim+self.objfeat_dim,\
            f"unimplement point dim: {data_start.shape[-1]}"
        
        if t is None:
            t = torch.randint(0, self.num_timesteps, size=(B,), device=self.device)
        else:
            assert t.shape == torch.Size([B])
        
        if noise is None:
            noise = torch.randn_like(data_start)
        else:
            assert noise.shape == data_start.shape 
            assert noise.dtype == data_start.dtype
            noise.to(self.device)

        # sample q(x_t | x_0)
        x_t = self.q_sample(x_0=data_start, t=t, eps=noise)

        # denoising net output
        denoise_out = denoise_fn(x_t, t, condition, condition_cross)
        assert denoise_out.shape == data_start.shape
        
        # if object class is knonw, we ignore all empty instances
        if self.class_dim == 1: # known class
            object_mask = data_start[:, :, self.bbox_dim] < 0
        else:
            object_mask = None
        
        if self.loss_type == 'mse':     # simplified loss
            loss_tensor = self.compute_mse_loss(
                data_start, x_t, t, denoise_out, noise=noise
            )
        elif self.loss_type == 'kl':    # exact kl-divergence
            loss_tensor = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, t=t, 
                condition=condition, condition_cross=condition_cross, 
            )
        else:
            raise NotImplemented
        
        # compute feature seperated loss for data logging
        feat_separated_losses = dict()
        for feat_name, (feat_dim, start_ind) in self.dim_dict.items():
            if feat_name in ['class', 'object'] and self.class_dim == 1:
                feat_loss = torch.zeros(B, device=self.device)
            else:
                feat_loss = self.mean_per_batch(
                    loss_tensor, feat_dim, start_ind, mask=object_mask
                )
            feat_separated_losses['loss.' + feat_name] = feat_loss
        
        # overall diffusin loss
        if self.loss_separate:
            losses = feat_separated_losses['loss.bbox'] + \
                     feat_separated_losses['loss.class'] + \
                     feat_separated_losses['loss.object'] + \
                     feat_separated_losses['loss.objfeat']
        else:
            losses = self.mean_per_batch(loss_tensor, mask=object_mask)
        losses_weight = losses * self._extract(self.loss_weight, t, losses.shape)
        
        # additional bounding box regularization loss on mean of p(x_{t-1} | x_t)
        if self.loss_iou:
            if self.model_mean_type == 'eps':
                x_recon = self._predict_start_from_eps(x_t, t, eps=denoise_out)
            elif self.model_mean_type == 'x0':
                x_recon = denoise_out
            elif self.model_mean_type == 'v':
                x_recon = self._predict_start_from_v(x_t, t, v=denoise_out)
            elif self.model_mean_type == 'mu':
                x_recon = self._predict_start_from_mu(x_t, t, mu=denoise_out)
            else:
                raise NotImplemented
            x_recon = self._clip_bbox(x_recon)
            
            loss_iou = self.bbox_iou_losses(x_recon, t)
            losses_weight += loss_iou
            feat_separated_losses['loss.liou'] = loss_iou
            
        assert losses_weight.shape == torch.Size([B])
        return (losses_weight.mean(),
                {k: v.mean() for k, v in feat_separated_losses.items()})

    def descale_to_origin(self, x, minimum, maximum):
        '''
            x shape : BxNx3
            minimum, maximum shape: 3
        '''
        x = (x + 1) / 2
        x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
        return x

    def bbox_iou_losses(self, x_recon, t):
        trans_recon = x_recon[:, :, 0:self.translation_dim]
        sizes_recon = x_recon[:, :, self.translation_dim:
                            self.translation_dim + self.size_dim]
        if self.objectness_dim > 0:
            start_index = self.bbox_dim + self.class_dim
            end_index = start_index + self.objectness_dim
            obj_recon = x_recon[:, :, start_index:end_index]
            valid_mask = (obj_recon >= 0).float().squeeze(2)
        else:
            start_index = self.bbox_dim + self.class_dim - 1
            end_index = self.bbox_dim + self.class_dim
            obj_recon = x_recon[:, :, start_index:end_index]
            valid_mask = (obj_recon <= 0).float().squeeze(2)
        # descale bounding box to world coordinate system
        descale_trans = self.descale_to_origin(
            trans_recon, self._centroids_min, self._centroids_max
        )
        descale_sizes = self.descale_to_origin(
            sizes_recon, self._sizes_min, self._sizes_max
        )
        # get the bbox corners
        axis_aligned_bbox_corn = torch.cat([
            descale_trans - descale_sizes / 2, 
            descale_trans + descale_sizes / 2
        ], dim=-1)
        assert axis_aligned_bbox_corn.shape[-1] == 6
        # compute iou - this uses axis-aligned bboxes and does not account for rotation
        bbox_iou = axis_aligned_bbox_overlaps_3d(
            axis_aligned_bbox_corn, axis_aligned_bbox_corn
        )
        bbox_iou_mask = valid_mask[:, :, None] * valid_mask[:, None, :]
        bbox_iou_valid = bbox_iou * bbox_iou_mask

        # get the iou loss weight w.r.t time
        w_iou = 0.1 * self._extract(self.alpha_bar, t, bbox_iou.shape)
        loss_iou_valid_avg = (w_iou * bbox_iou_valid).sum(dim=[1, 2]) \
                / (bbox_iou_mask.sum(dim=[1, 2]) + 1e-6)
        return loss_iou_valid_avg
    

class DiffusionPoint(nn.Module):
    def __init__(self, denoise_net:nn.Module, network_dim, schedule_type='linear',
                 beta_start=0.0001, beta_end=0.02, time_num=1000, warm_ratio=0.1,
                 model_mean_type='eps', model_var_type ='fixedsmall',
                 loss_type='mse', loss_separate=True, loss_iou=False,
                 train_stats_file=None):
        super(DiffusionPoint, self).__init__()
        
        betas = get_betas(schedule_type, beta_start, beta_end, time_num, warm_ratio)
        self.diffusion = GaussianDiffusion(
            network_dim, betas, loss_type, model_mean_type, model_var_type, 
            loss_separate, loss_iou, train_stats_file
        )
        
        self.model = denoise_net
    
    def _denoise(self, data, t, condition, condition_cross):
        B, N, D = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        denoising_channels = getattr(self.model, "channels", D)
        if D != denoising_channels:
            class_dim = denoising_channels - D
            out = self.model(
                torch.cat([data, condition[..., :class_dim]], dim=-1), t,
                condition[..., class_dim:], condition_cross
            )
            return out[..., :D]
        else:
            out = self.model(data, t, condition, condition_cross)
            assert out.shape == torch.Size([B, N, D])
            return out

    def get_loss_iter(self, data, noise=None, condition=None, 
                      condition_cross=None):
        device = data.device
        self.diffusion._move_tensors(device)
        self.model.to(device)

        losses, loss_dict = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, noise=noise, 
            condition=condition, condition_cross=condition_cross
        )
        return losses, loss_dict
    
    def gen_samples(self, shape, device, condition=None, condition_cross=None,
                    target_data=None, target_mask=None, freq=None, clip_denoised=False):
        self.diffusion._move_tensors(device)

        if target_data is not None:
            assert target_mask is not None
            target_fn = lambda x: x * (~target_mask) + target_data * target_mask
        else:
            target_fn = lambda x: x
        
        x_end = torch.randn(size=shape, dtype=torch.float, device=device)
        return self.diffusion.p_sample_loop(
            self._denoise, x_end=x_end, x_target_fn=target_fn,
            condition=condition, condition_cross=condition_cross,
            log_freq=freq, clip_denoised=clip_denoised
        )
