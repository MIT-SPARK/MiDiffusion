#
# Modified from: 
#   https://github.com/cientgu/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/transformer_utils.py
#

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from .time_embedding import SinusoidalPosEmb

LAYER_NROM_EPS = 1e-5   # pytorch's default: 1e-5


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class _AdaNorm(nn.Module):
    """Base normalization layer that incorporate timestep embeddings"""
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str="adalayernorm_abs"
    ):
        super().__init__()
        assert n_embd % 2 == 0
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(n_embd, num_steps=max_timestep)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    """Norm layer modified to incorporate timestep embeddings"""
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str="adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, eps=LAYER_NROM_EPS, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: Tensor):
        # (B, N, n_embd),(B,) -> (B, N, n_embd)
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)   # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    """Base instance normalization layer that incorporate timestep embeddings"""
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str="adainsnorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd, eps=LAYER_NROM_EPS)

    def forward(self, x: Tensor, timestep: Tensor):
        # (B, N, n_embd),(B,) -> (B, N, n_embd)
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)   # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale) \
            + shift
        return x


class SelfAttention(nn.Module):
    """Multi-head self attention block - in transformer encoder"""
    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.mha = nn.MultiheadAttention(
            n_embd, n_head, dropout, batch_first=batch_first
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        return self.mha(
            x, x, x, need_weights=False,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]


class CrossAttention(nn.Module):
    """Multi-head cross attention block - in transformer decoder"""
    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True, kv_embd=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.mha = nn.MultiheadAttention(
            n_embd, n_head, dropout, batch_first=batch_first, kdim=kv_embd, vdim=kv_embd
        )

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        return self.mha(
            query=q, key=kv, value=kv, need_weights=False,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]
    

class Block(nn.Module):
    """ Time-conditioned transformer block """
    def __init__(self,
                 n_embd=512,
                 n_head=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activate='GELU',
                 num_timesteps=1000,
                 timestep_type='adalayernorm_abs',
                 attn_type='self',
                 num_labels=None,           # attn_type = 'selfcondition'
                 label_type='adalayernorm', # attn_type = 'selfcondition'
                 cond_emb_dim=None,         # attn_type = 'selfcross'
                 mlp_type = 'fc',
                 ):
        super().__init__()
        self.attn_type = attn_type

        if "adalayernorm" in timestep_type:
            self.ln1 = AdaLayerNorm(n_embd, num_timesteps, timestep_type)
        elif "adainnorm" in timestep_type:
            self.ln1 = AdaInsNorm(n_embd, num_timesteps, timestep_type)
        else:
            raise ValueError(f"timestep_type={timestep_type} not valid.")
        
        if attn_type == 'self':
            self.attn = SelfAttention(
                n_embd=n_embd, n_head=n_head, dropout=dropout
            )
            self.ln2 = nn.LayerNorm(n_embd)
        elif attn_type == 'selfcondition':  # conditioned on int labels
            self.attn = SelfAttention(
                n_embd=n_embd, n_head=n_head, dropout=dropout
            )
            if 'adalayernorm' in label_type:
                self.ln2 = AdaLayerNorm(n_embd, num_labels, label_type)
            else:
                self.ln2 = AdaInsNorm(n_embd, num_labels, label_type)
        elif attn_type == 'selfcross':      # cross attention with cond_emb
            self.attn1 = SelfAttention(
                n_embd=n_embd, n_head=n_head, dropout=dropout
            )
            self.attn2 = CrossAttention(
                n_embd=n_embd, n_head=n_head, dropout=dropout, kv_embd=cond_emb_dim,
            )
            if 'adalayernorm' in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, num_timesteps, timestep_type)
            else:
                raise ValueError(f"timestep_type={timestep_type} not valid.")
            self.ln2 = nn.LayerNorm(n_embd)
        else:
            raise ValueError(f"attn_type={attn_type} not valid.")
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'fc':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, dim_feedforward),
                act,
                nn.Linear(dim_feedforward, n_embd),
                nn.Dropout(dropout),
            )
        else:
            raise NotImplemented

    def forward(self, x, timestep, cond_output=None, mask=None):    
        if self.attn_type == "self":
            x = x + self.attn(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.mlp(self.ln2(x))
        elif self.attn_type == "selfcondition":
            x = x + self.attn(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.mlp(self.ln2(x, cond_output))
        elif self.attn_type == "selfcross":
            x = x + self.attn1(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.attn2(self.ln1_1(x, timestep), cond_output, attn_mask=mask)
            x = x + self.mlp(self.ln2(x))
        else:
            return NotImplemented
        return x


class DenoiseTransformer(nn.Module):
    """Base denoising transformer class"""
    def __init__(
        self,
        n_layer=4,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate='GELU',
        num_timesteps=1000,
        timestep_type='adalayernorm_abs',
        context_dim=256,
        mlp_type='fc',
    ):
        super().__init__()

        # transformer backbone
        if context_dim == 0:
            self.tf_blocks = nn.Sequential(*[Block(
                n_embd, n_head, dim_feedforward, dropout, activate,
                num_timesteps, timestep_type, mlp_type=mlp_type,
                attn_type='self',
            ) for _ in range(n_layer)])
        else:
            self.tf_blocks = nn.Sequential(*[Block(
                n_embd, n_head, dim_feedforward, dropout, activate,
                num_timesteps, timestep_type, mlp_type=mlp_type,
                attn_type='selfcross', cond_emb_dim=context_dim,
            ) for _ in range(n_layer)])
    
    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size*2),
                nn.GELU(),
                nn.Linear(hidden_size*2, hidden_size),
            ]
        return nn.Sequential(*mlp_layers)
    
    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        ]
        return nn.Sequential(*mlp_layers)
