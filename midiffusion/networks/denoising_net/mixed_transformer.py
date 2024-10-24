import torch
from torch import nn
from .transformer_utils import DenoiseTransformer


class MixedDenoiseTransformer(DenoiseTransformer):
    def __init__(
        self,
        network_dim,
        seperate_all=True,
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
        concate_features=False,
    ):
        # initialize self.tf_blocks, the transformer backbone
        super().__init__(
            n_layer, n_embd, n_head, dim_feedforward, dropout, activate, 
            num_timesteps, timestep_type, context_dim, mlp_type
        )
        assert network_dim["class_dim"] > 0
        
        # feature dimensions
        self.objectness_dim, self.class_dim, self.objfeat_dim = \
            network_dim["objectness_dim"], network_dim["class_dim"], \
                network_dim["objfeat_dim"]
        self.translation_dim, self.size_dim, self.angle_dim = \
            network_dim["translation_dim"], network_dim["size_dim"], \
                network_dim["angle_dim"]
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.geo_dim = self.bbox_dim + self.objectness_dim + self.objfeat_dim

        # Feature specific processing
        self.concate_features = concate_features
        if concate_features:
            n_features = 2 + (self.objectness_dim > 0) + (self.objectness_dim > 0)
            feat_embd = n_embd // n_features
            geo_embd = feat_embd * (n_features - 1)
            class_embd = n_embd - geo_embd
            decode_embd = n_embd + network_dim["class_dim"] # geometric decoder
            print("concatenate features (class embd: {}, geometric embd: {})"
                  .format(class_embd, feat_embd))
        else:
            feat_embd = n_embd
            geo_embd = n_embd
            class_embd = n_embd
            decode_embd = n_embd
        
        # semantic feature - add additional [mask] embedding
        self.class_emb = nn.Embedding(network_dim["class_dim"] + 1, class_embd)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, network_dim["class_dim"]),
        )
        
        # geometric features
        self.seperate_all = seperate_all
        if self.seperate_all:
            self.bbox_embedf = self._encoder_mlp(feat_embd, self.bbox_dim)
            self.bbox_hidden2output = self._decoder_mlp(decode_embd, self.bbox_dim)            
            if self.objectness_dim > 0:
                self.objectness_embedf = self._encoder_mlp(feat_embd, self.objectness_dim)
                self.objectness_hidden2output = self._decoder_mlp(decode_embd, self.objectness_dim)
            if self.objfeat_dim > 0:
                self.objfeat_embedf = self._encoder_mlp(feat_embd, self.objfeat_dim)
                self.objfeat_hidden2output = self._decoder_mlp(decode_embd, self.objfeat_dim)
        else:
            self.init_mlp = self._encoder_mlp(geo_embd, self.geo_dim)
            self.geo_hidden2output = self._decoder_mlp(decode_embd, self.geo_dim)
    
    def forward(self, x_semantic, x_geometry, time, context=None, context_cross=None): 
        # x: (B, N, C)
        if context_cross is not None:
            raise NotImplemented    # TODO

        # initial processing
        x_class = self.class_emb(x_semantic)
        if self.seperate_all:
            x_geo = self.bbox_embedf(x_geometry[:, :, 0:self.bbox_dim])
            if self.objectness_dim > 0:
                start_index = self.bbox_dim
                x_object = self.objectness_embedf(
                    x_geometry[:, :, start_index:start_index+self.objectness_dim]
                )
                if self.concate_features:
                    x_geo = torch.cat([x_geo, x_object], dim=2).contiguous()
                else:
                    x_geo += x_object
            if self.objfeat_dim > 0:
                start_index = self.bbox_dim+self.objectness_dim
                x_objfeat = self.objfeat_embedf(
                    x_geometry[:, :, start_index:start_index+self.objfeat_dim]
                )
                if self.concate_features:
                    x_geo = torch.cat([x_geo, x_objfeat], dim=2).contiguous()
                else:
                    x_geo += x_objfeat
        else:
            x_geo = self.init_mlp(x_geometry)
        
        if self.concate_features:
            x = torch.cat([x_class, x_geo], dim=2).contiguous()
        else:
            x = x_class + x_geo

        # transformer
        for block_idx in range(len(self.tf_blocks)):
            x = self.tf_blocks[block_idx](x, time, context)

        # final processing
        out_class = self.to_logits(x)
        if self.concate_features:
            x = torch.cat([x, out_class], dim=2).contiguous()
        if self.seperate_all:
            out  = self.bbox_hidden2output(x)
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=2).contiguous()
            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=2).contiguous()
        else:
            out = self.geo_hidden2output(x)
        
        return out_class, out
