#
# Modified from:
#   https://github.com/tangjiapeng/DiffuScene
#

from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from .time_embedding import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb

# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResidualCross(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, context, *args, **kwargs):
        return self.fn(x, context, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    if dim_out is None or dim == dim_out:
        return nn.Identity()
    else:
        return nn.Sequential(
            #nn.Upsample(scale_factor = 2, mode = 'nearest'),

            nn.Conv1d(dim, default(dim_out, dim), 1)
        )


def Downsample(dim, dim_out=None):
    if dim_out is None or dim == dim_out:
        return nn.Identity()
    else:
        return nn.Sequential(
            #return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)
        
            nn.Conv1d(dim, default(dim_out, dim), 1) 
        )

# ResNet block

class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 1, padding=0) # 3-->1
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            if len(time_emb.shape) == 2:
                time_emb = rearrange(time_emb, 'b c -> b c 1')
            else:
                time_emb = rearrange(time_emb, 'b n c -> b c n')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


# Attention module
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv
        )

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        # sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class LinearAttentionCross(nn.Module):
    def __init__(self, dim, context_dim=None, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if context_dim is None:
            context_dim = dim
        self.to_q  = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_kv = nn.Conv1d(context_dim, hidden_dim*2, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, context):
        b, c, n = x.shape
        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = 1)
        q = rearrange(q, 'b (h c) n -> b h c n', h = self.heads)
        k, v = map(
            lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), kv
        )

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)


# Group normalization

class LayerNorm(nn.Module):     # similar to nn.GroupNorm without bias param
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    

class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context):
        x = self.norm(x)
        return self.fn(x, context)


# Conditional U-Net

class Unet1D(nn.Module):
    def __init__(
        self,
        network_dim,
        dim=256,
        init_dim = None,                    # default: dim
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,                       # ignored if seperate_all=True
        seperate_all=False,
        context_dim = 256,
        cross_condition=False,
        cross_condition_dim=256,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # model flags
        self.cross_condition = cross_condition
        self.seperate_all = seperate_all

        # feature dimensions
        self.objectness_dim, self.class_dim, self.objfeat_dim = \
            network_dim["objectness_dim"], network_dim["class_dim"], \
                network_dim["objfeat_dim"]
        self.translation_dim, self.size_dim, self.angle_dim = \
            network_dim["translation_dim"], network_dim["size_dim"], \
                network_dim["angle_dim"]
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.context_dim = context_dim
        if cross_condition:
            self.cross_condition_dim = cross_condition_dim

        # Initial feature specific processing     
        if self.seperate_all:
            self.bbox_embedf = Unet1D._encoder_mlp(dim, self.bbox_dim)
            self.bbox_hidden2output = Unet1D._decoder_mlp(dim, self.bbox_dim)
            feature_str = "translation/size/angle"
            
            if self.class_dim > 0:
                self.class_embedf = Unet1D._encoder_mlp(dim, self.class_dim)
                feature_str += "/class"
            if self.objectness_dim > 0:
                self.objectness_embedf = Unet1D._encoder_mlp(dim, self.objectness_dim)
                feature_str += "/objectness"
            if self.objfeat_dim > 0:
                self.objfeat_embedf = Unet1D._encoder_mlp(dim, self.objfeat_dim)
                feature_str += "/objfeat"

            input_channels = dim
            print('separate unet1d encoder/decoder of {}'.format(feature_str))
        else:
            input_channels = channels
            print('unet1d encoder of all object properties')

        # U-Net initialization
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1) #nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=context_dim), 
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                ResidualCross(PreNormCross(
                    dim_in, LinearAttentionCross(dim_in, cross_condition_dim)
                )) if cross_condition else nn.Identity(),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last 
                else nn.Conv1d(dim_in, dim_out, 1) #3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block0 = block_klass(mid_dim, mid_dim, time_emb_dim=context_dim) 
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn_cross = ResidualCross(PreNormCross(
            mid_dim, LinearAttentionCross(mid_dim, cross_condition_dim)
        )) if cross_condition else nn.Identity()
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_in, time_emb_dim=context_dim), 
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResidualCross(PreNormCross(
                    dim_out, LinearAttentionCross(dim_out, cross_condition_dim)
                )) if cross_condition else nn.Identity(),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last 
                else nn.Conv1d(dim_out, dim_in, 1) #3, padding = 1)
            ]))

        self.final_res_block = block_klass(init_dim * 2, dim, time_emb_dim=time_dim)

        # Final feature specific processing     
        if self.seperate_all:
            self.bbox_hidden2output = Unet1D._decoder_mlp(dim, self.bbox_dim)
            
            if self.class_dim > 0:
                self.class_hidden2output = Unet1D._decoder_mlp(dim, self.class_dim)
            if self.objectness_dim > 0:
                self.objectness_hidden2output = Unet1D._decoder_mlp(dim, self.objectness_dim)
            if self.objfeat_dim > 0:
                self.objfeat_hidden2output = Unet1D._decoder_mlp(dim, self.objfeat_dim)
            
        else:
            default_out_dim = channels * (1 if not learned_variance else 2)
            self.out_dim = default(out_dim, default_out_dim)
            self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
        
    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
                nn.Conv1d(input_size, hidden_size, 1),
                nn.GELU(),
                nn.Conv1d(hidden_size, hidden_size*2, 1),
                nn.GELU(),
                nn.Conv1d(hidden_size*2, hidden_size, 1),
            ]
        return nn.Sequential(*mlp_layers)
    
    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Conv1d(hidden_size, hidden_size*2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size*2, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, output_size, 1),
        ]
        return nn.Sequential(*mlp_layers)
    

    def forward(self, x, time, context=None, context_cross=None): 
        # (B, N, C) --> (B, C, N)
        x = torch.permute(x, (0, 2, 1)).contiguous()
        if context_cross is not None:
            # [B, N, C] --> [B, C, N]
            context_cross = torch.permute(context_cross, (0, 2, 1)).contiguous()

        # initial processing
        if self.seperate_all:
            x_bbox = self.bbox_embedf(x[:, 0:self.bbox_dim, :])

            if self.class_dim > 0:
                start_index = self.bbox_dim
                x_class = self.class_embedf(
                    x[:, start_index:start_index+self.class_dim, :]
                )
            else:
                x_class = 0
            
            if self.objectness_dim > 0:
                start_index = self.bbox_dim+self.class_dim
                x_object = self.objectness_embedf(
                    x[:, start_index:start_index+self.objectness_dim, :]
                )
            else:
                x_object = 0
            
            if self.objfeat_dim > 0:
                start_index = self.bbox_dim+self.class_dim+self.objectness_dim
                x_objfeat = self.objfeat_embedf(
                    x[:, start_index:start_index+self.objfeat_dim, :]
                )
            else:
                x_objfeat = 0
                
            x = x_bbox + x_class + x_object + x_objfeat

        # unet-1D 
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) 

        h = []

        for block0, block1, attncross, block2, attn, downsample in self.downs:
            x = block0(x, context) 
            x = block1(x, t)
            h.append(x)

            x = attncross(x, context_cross) if self.cross_condition else attncross(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block0(x, context)
        x = self.mid_block1(x, t)
        x = self.mid_attn_cross(x, context_cross) if self.cross_condition else self.mid_attn_cross(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block0, block1, attncross, block2, attn, upsample in self.ups:
            x = block0(x, context) 
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = attncross(x, context_cross) if self.cross_condition else attncross(x)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
 
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)

        # final processing
        if self.seperate_all:
            out  = self.bbox_hidden2output(x)
            if self.class_dim > 0:
                out_class = self.class_hidden2output(x)
                out = torch.cat([out, out_class], dim=1).contiguous()
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=1).contiguous()
            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=1).contiguous()
        else:
            out = self.final_conv(x)
        
        # (B, N, C) <-- (B, C, N)
        out = torch.permute(out, (0, 2, 1)).contiguous()
        return out