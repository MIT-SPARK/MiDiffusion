#
# Modified from: 
#   https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/diffusion_transformer.py
#

import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .diffusion_base import BaseDiffusion, EPS_PROB, LOG_ZERO
from .denoising_net.mixed_transformer import MixedDenoiseTransformer


'''
helper functions
'''

def log_1_min_a(a):
    """log(1 - exp(a))"""
    return torch.log(1 - a.exp() + EPS_PROB)


def log_add_exp(a, b):
    """log(exp(a) + exp(b))"""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    """convert (*, N) index tensor to (*, C, N) log 1-hot tensor"""
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, x.ndim))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=EPS_PROB))
    return log_x


def log_onehot_to_index(log_x):
    """argmax(log_x, dim=1)"""
    return log_x.argmax(1)


def alpha_schedule(
    num_timesteps, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999
):
    # note: 0.0 will tends to raise unexpected behaviour (e.g., log(0.0)), thus avoid 0.0
    assert att_1 > 0.0 and att_T > 0.0 and ctt_1 > 0.0 and ctt_T > 0.0
    assert att_1 + ctt_1 <= 1.0 and att_T + ctt_T <= 1.0

    att = np.arange(0, num_timesteps) / (num_timesteps - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, num_timesteps) / (num_timesteps - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N

    def _f(x):
        return torch.tensor(x.astype("float64"))

    return _f(at), _f(bt), _f(ct), _f(att), _f(btt), _f(ctt)

'''
model
'''

class MaskAndReplaceDiffusion(BaseDiffusion):
    def __init__(self, num_classes, noise_params, model_output_type="x0", 
                 mask_weight=1, auxiliary_loss_weight=0, 
                 adaptive_auxiliary_loss=False):
        super().__init__()
        
        assert model_output_type in ["x0", "x_prev"]
        assert auxiliary_loss_weight >= 0
        assert mask_weight >= 0
        self.num_classes = num_classes  # TODO: currently, this is includes 'empty' and 'mask'
        self.model_output_type = model_output_type
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight  # 'mask' token weight

        # diffusion noise params
        at, bt, ct, att, btt, ctt = noise_params
        assert at.shape[0] == bt.shape[0] == ct.shape[0]
        assert att.shape[0] == btt.shape[0] == ctt.shape[0] == at.shape[0] + 1   
        self.num_timesteps = at.shape[0]

        log_at, log_bt, log_ct = torch.log(at), torch.log(bt), torch.log(ct)
        log_cumprod_at, log_cumprod_bt, log_cumprod_ct = \
            torch.log(att), torch.log(btt), torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item()
            < 1.0e-5
        )

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t_1, t):
        """
        log(Q_t * exp(log_x_t_1)), diffusion step: q(x_t | x_{t-1})
        """
        # log_x_t_1 (B, C, N)
        log_at = self._extract(self.log_at, t, log_x_t_1.shape)  # at
        log_bt = self._extract(self.log_bt, t, log_x_t_1.shape)  # bt
        log_ct = self._extract(self.log_ct, t, log_x_t_1.shape)  # ct
        log_1_min_ct = self._extract(self.log_1_min_ct, t, log_x_t_1.shape)  # 1-ct

        log_probs = torch.cat([
                log_add_exp(log_x_t_1[:, :-1, :] + log_at, log_bt),   # dropped a small term
                log_add_exp(log_x_t_1[:, -1:, :] + log_1_min_ct, log_ct),
            ], dim=1)

        return log_probs
    
    def q_pred(self, log_x_start, t):
        """
        log(bar{Q}_t * exp(log_x_start)), diffuse the data to time t: q(x_t | x_0)
        """
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = self._extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = self._extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = self._extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = self._extract(
            self.log_1_min_cumprod_ct, t, log_x_start.shape
        )  # 1-ct~

        log_probs = torch.cat([
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),  # simplified
            ], dim=1)

        return log_probs
    
    def q_posterior(self, log_x_start, log_x_t, t):
        """
        log of prosterior probability q(x_{t-1}|x_t,x_0')
        """
        B, C, N = log_x_start.shape
        log_one_vector = torch.zeros(B, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.full((B, 1, N), LOG_ZERO).type_as(log_x_t)
        
        # notice that log_x_t is onehot
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = self._extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = self._extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = \
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, LOG_ZERO, 0)
    
    @staticmethod
    def log_pred_from_denoise_out(denoise_out):
        """
        convert output of denoising network to log probability over classes and [mask]
        """
        out = denoise_out.permute((0, 2, 1))     # (B, N, C-1) -> (B, C-1, N)
        B, _, N = out.shape

        log_pred = F.log_softmax(out.double(), dim=1).float()
        log_pred = torch.clamp(log_pred, LOG_ZERO, 0)
        log_zero_vector = torch.full((B, 1, N), LOG_ZERO).type_as(log_pred)
        return torch.cat((log_pred, log_zero_vector), dim=1)
    
    def predict_denoise(self, denoise_fn, log_x_t, t, condition=None, 
                        condition_cross=None):
        """
        compute denoise_fn(data, t, condition, condition_cross) and convert output to log prob
        """
        x_t = log_onehot_to_index(log_x_t)  # (B, N)
        out = denoise_fn(x_t, t, condition, condition_cross)
        log_pred = self.log_pred_from_denoise_out(out)
        assert log_pred.shape == log_x_t.shape

        return log_pred
    
    def p_pred(self, denoise_fn, log_x_t, t, condition=None, condition_cross=None):             
        """
        log denoising probability, denoising step: p(x_{t-1} | x_t)
        """
        if self.model_output_type == 'x0':
            # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
            log_x_recon = self.predict_denoise(
                denoise_fn, log_x_t, t, condition, condition_cross
            )
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x_t, t=t
            )
            return log_model_pred, log_x_recon
        elif self.model_output_type == 'x_prev':
            log_model_pred = self.predict_denoise(
                denoise_fn, log_x_t, t, condition, condition_cross
            )
            return log_model_pred, None
        else:
            raise NotImplemented
    
    '''
    sampling
    '''

    def q_sample_one_step(self, log_x_t_1, t, no_mask=False):
        """
        sample from q(x_t | x_{t-1})
        """
        log_EV_qxt = self.q_pred_one_timestep(log_x_t_1, t)
        log_sample = self.log_sample_categorical(log_EV_qxt, no_mask)
        return log_sample

    def q_sample(self, log_x_start, t, no_mask=False):
        """
        sample from q(x_t | x_0)
        """
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0, no_mask)
        return log_sample
    
    @torch.no_grad()
    def p_sample(self, denoise_fn, log_x_t, t, condition, condition_cross=None):               
        """
        sample x_{t-1} from p(x_{t-1} | x_t)
        """
        model_log_prob, _ = self.p_pred(denoise_fn, log_x_t, t, condition, condition_cross)
        log_sample = self.log_sample_categorical(model_log_prob)
        return log_sample
    
    def log_sample_categorical(self, logits, no_mask=False):
        """
        sample from log probability under gumbel noise, return results as log of 1-hot embedding
        (no_mask=True means sampling without the last [mask] class)
        """
        # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + EPS_PROB) + EPS_PROB)
        if no_mask:
            sample = (gumbel_noise + logits)[:, :-1, :].argmax(dim=1)
        else:
            sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
        else:
            raise ValueError
        return t, pt
        
    '''
    loss
    '''
    def compute_kl_loss(self, log_x_start, log_x_t, t, log_pred_prob):
        """compute train loss of each variable"""
        log_q_prob = self.q_posterior(log_x_start, log_x_t, t)
        kl = self.multinomial_kl(log_q_prob, log_pred_prob)
        decoder_nll = -log_categorical(log_x_start, log_pred_prob)

        t0_mask = (t == 0).unsqueeze(1).repeat(1, log_x_start.shape[-1])
        kl_loss = torch.where(t0_mask, decoder_nll, kl)
        return kl_loss

    def compute_aux_loss(self, log_x_start, log_x0_recon, t):
        """compute auxilary loss regulating predicted x0"""
        aux_loss = self.multinomial_kl(
            log_x_start[:, :-1, :], log_x0_recon[:, :-1, :]
        )

        t0_mask = (t == 0).unsqueeze(1).repeat(1, log_x_start.shape[-1])
        aux_loss = torch.where(t0_mask, torch.zeros_like(aux_loss), aux_loss)
        return aux_loss

    def p_losses(self, denoise_fn, x_start, t=None, pt=None, condition=None):
        assert self.model_output_type == 'x0'
        if t is None or pt is None:
            t, pt = self.sample_time(x_start.size(0), x_start.device, "uniform")
        
        log_xstart = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_xstart, t=t)

        log_model_prob, log_x0_recon = self.p_pred(denoise_fn, log_xt, t, condition)

        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size(0)):
            this_t = t[index].item()
            same_rate = \
                (x0_recon[index] == x0_real[index]).sum().cpu() / x0_real.size(1)
            self.diffusion_acc_list[this_t] = \
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            same_rate = \
                (xt_1_recon[index] == xt_recon[index]).sum().cpu() / xt_recon.size(1)
            self.diffusion_keep_list[this_t] = \
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9

        # Compute train loss
        loss_tensor = self.compute_kl_loss(log_xstart, log_xt, t, log_model_prob)
        if self.mask_weight != 1:   # adjust [mask] token weight
            mask_region = (log_onehot_to_index(log_xt) == self.num_classes - 1)
            loss_tensor = torch.where(
                mask_region, self.mask_weight * loss_tensor, loss_tensor
            )
        kl_loss = self.sum_last_dims(loss_tensor, keep_dims=1)
        
        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))
        
        # Upweigh loss term of the kl
        loss1 = kl_loss / pt
        vb_loss = loss1.mean()
        losses_dict = {"kl_loss": loss1.mean()}

        if self.auxiliary_loss_weight > 0:
            loss_tensor = self.compute_aux_loss(log_xstart, log_x0_recon, t)
            if self.mask_weight != 1:   # adjust [mask] token weight
                loss_tensor = torch.where(
                    mask_region, self.mask_weight * loss_tensor, loss_tensor
                )
            aux_loss = self.sum_last_dims(loss_tensor, keep_dims=1)
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * aux_loss / pt
            losses_dict["aux_loss"] = loss2.mean()
            vb_loss += loss2.mean()

        return vb_loss, losses_dict
    
    def p_sample_loop(self, denoise_fn, log_x_end, condition, condition_cross=None,
                      sample_freq=None):
        B, C, N = log_x_end.shape
        assert C == self.num_classes
        if sample_freq:
            pred_traj = [log_onehot_to_index(log_x_end)]
        
        log_x_t = log_x_end
        total_steps = self.num_timesteps
        for t in reversed(range(0, total_steps)):
            t_ = torch.full((B,), t, dtype=torch.int64, device=self.device)
            log_x_t = self.p_sample(
                denoise_fn=denoise_fn, log_x_t=log_x_t, t=t_, 
                condition=condition, condition_cross=condition_cross
            )     # log_x_t is log_onehot
            if sample_freq and (t % sample_freq == 0 or t == total_steps - 1):
                pred_traj.append(log_onehot_to_index(log_x_t))

        if sample_freq:
            return pred_traj
        else:
            return log_onehot_to_index(log_x_t)


class DiscreteDiffusionPoint(nn.Module):
    def __init__(self, denoise_net:nn.Module, class_dim, time_num=1000, 
                 model_output_type="x0", mask_weight=1, auxiliary_loss_weight=0, 
                 adaptive_auxiliary_loss=False, **kwargs):
          
        super(DiscreteDiffusionPoint, self).__init__()
        
        noise_params = alpha_schedule(time_num, class_dim, **kwargs)
        
        self.diffusion = MaskAndReplaceDiffusion(
            class_dim + 1, noise_params, model_output_type, 
            mask_weight, auxiliary_loss_weight, adaptive_auxiliary_loss
        )
        self.model = denoise_net
        assert self.diffusion.num_classes - 1 == self.model.class_dim

    def _denoise(self, data, t, condition, condition_cross=None):
        B, N = data.shape
        C = self.diffusion.num_classes - 1
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64
        
        if isinstance(self.model, MixedDenoiseTransformer):
            out, _ = self.model(
                x_semantic=data, x_geometry=None, time=t, 
                context=condition, context_cross=condition_cross
            )
        else:
            out = self.model(data, t, condition)
        assert out.shape == torch.Size([B, N, C]), f"Error: {out.shape} != {B, N, C}"
        return out

    def get_loss_iter(self, data, condition=None):
        device = data.device
        self.diffusion._move_tensors(device)
        self.model.to(device)

        losses, loss_dict = self.diffusion.p_losses(
            denoise_fn=self._denoise, x_start=data, condition=condition,
        )
        return losses, loss_dict
    
    def gen_samples(self, shape, device, condition, condition_cross=None,
                    freq=None):
        B, N = shape
        self.diffusion._move_tensors(device)
        
        log_zeros = torch.full((B, self.diffusion.num_classes-1, N), LOG_ZERO, device=device)
        log_ones = torch.zeros((B, 1, N), device=device)
        log_x_end = torch.cat((log_zeros, log_ones), dim=1)

        return self.diffusion.p_sample_loop(
            self._denoise, log_x_end=log_x_end, 
            condition=condition, condition_cross=condition_cross, 
            sample_freq=freq
        )
