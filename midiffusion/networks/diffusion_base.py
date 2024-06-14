import torch.nn as nn
import torch.utils.data

EPS_PROB = 1e-30    # minimum probability to make sure log prob is numerically stable
LOG_ZERO = -69      # substitute of log(0)


class BaseDiffusion(nn.Module):
    """
    Base class for diffusion model.
    Note that theoretically t = 1, ..., num_steps, the function argument t 
    starts from 0 (i.e. off by 1) due to python indexing.
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def q_pred(self, x_0, t):
        """
        Compute probability q(x_t | x_0)
        """
        raise NotImplementedError

    def q_sample(self, x_0, t):
        """
        Diffuse the data, i.e. sample from q(x_t | x_0)
        """
        raise NotImplementedError
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute posterior probability q(x_{t-1} | x_t, x_0)
        """
        raise NotImplementedError
    
    def p_pred(self, denoise_fn, x_t, t, condition, **kwargs):
        """
        Compute denoising probability p(x_{t-1} | x_t)
        """
        raise NotImplementedError

    def p_sample(self, denoise_fn, x_t, t, condition, **kwargs):
        """
        Denoise the data, i.e. sample from p(x_{t-1} | x_t)
        """
        raise NotImplementedError

    def p_sample_loop(self, denoise_fn, shape, condition, sample_freq=None, **kwargs):
        """
        Generate data by denoising recursively
        """
        raise NotImplementedError

    def p_losses(self, denoise_fn, x_0, condition, **kwargs):
        """
        Training loss calculation
        """
        raise NotImplementedError
    
    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def _move_tensors(self, device):
        """
        Move pre-computed parameters to specified device
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        self.device = torch.device(device)

    @staticmethod
    def sum_last_dims(data_tensor, keep_dims=1):
        """
        Sum over the last dimensions. Default: sum input data over each batch.
        """
        return data_tensor.reshape(*data_tensor.shape[:keep_dims], -1).sum(-1)
    
    @staticmethod
    def mean_per_batch(data_tensor, feat_size=None, start_ind=0, mask=None):
        """
        Given B x N x C input data, return a 1-D average tensor of size B,
        with option to specify a B x N mask or a feature range to select data.
        """
        B, N, C = data_tensor.shape

        # Handle the case where feat_size is zero
        if feat_size == 0:
            return torch.zeros(B, device=data_tensor.device)
        
        # Determine feature size if not specified
        if feat_size is None:
            feat_size = C - start_ind

        # Select the relevant feature range from the data
        data_selected = data_tensor[:, :, start_ind:start_ind + feat_size]

        # Compute mean with or without mask
        if mask is not None:
            assert mask.shape == (B, N) and (mask.sum(dim=1) > 0).all()
            masked_sum = (data_selected * mask.unsqueeze(-1)).sum(dim=[1, 2])
            return masked_sum / (mask.sum(dim=1) * feat_size)
        else:
            return data_selected.mean(dim=[1, 2])
