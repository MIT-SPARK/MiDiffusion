import math
import torch
from torch import nn, Tensor
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    """https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/transformer_utils.py"""
    def __init__(self, dim: int, num_steps: int=4000, rescale_steps: int=4000):
        super().__init__()
        self.dim = dim
        if num_steps != rescale_steps:
            self.num_steps = float(num_steps)
            self.rescale_steps = float(rescale_steps)
            self.input_scaling = True
        else:
            self.input_scaling = False

    def forward(self, x: Tensor):
        # (B) -> (B, self.dim)
        if self.input_scaling:
            x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
