import torch
from .transformer_utils import DenoiseTransformer


class ContinuousDenoiseTransformer(DenoiseTransformer):
    """Continuous denoising transformer network where all object properties are 
    treated as continuous"""
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
    ):
        # initialize self.tf_blocks, the transformer backbone
        super().__init__(
            n_layer, n_embd, n_head, dim_feedforward, dropout, activate, 
            num_timesteps, timestep_type, context_dim, mlp_type
        )
        
        # feature dimensions
        self.objectness_dim, self.class_dim, self.objfeat_dim = \
            network_dim["objectness_dim"], network_dim["class_dim"], \
                network_dim["objfeat_dim"]
        self.translation_dim, self.size_dim, self.angle_dim = \
            network_dim["translation_dim"], network_dim["size_dim"], \
                network_dim["angle_dim"]
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.channels = self.bbox_dim + self.objectness_dim + self.class_dim + self.objfeat_dim

        # Initial feature specific processing     
        self.seperate_all = seperate_all
        if self.seperate_all:
            self.bbox_embedf = self._encoder_mlp(n_embd, self.bbox_dim)
            self.bbox_hidden2output = self._decoder_mlp(n_embd, self.bbox_dim)
            feature_str = "translation/size/angle"
            
            if self.class_dim > 0:
                self.class_embedf = self._encoder_mlp(n_embd, self.class_dim)
                feature_str += "/class"
            if self.objectness_dim > 0:
                self.objectness_embedf = self._encoder_mlp(n_embd, self.objectness_dim)
                feature_str += "/objectness"
            if self.objfeat_dim > 0:
                self.objfeat_embedf = self._encoder_mlp(n_embd, self.objfeat_dim)
                feature_str += "/objfeat"
            print('separate unet1d encoder/decoder of {}'.format(feature_str))
        else:
            self.init_mlp = self._encoder_mlp(n_embd, self.channels)
            print('unet1d encoder of all object properties')

        # Final feature specific processing     
        if self.seperate_all:
            self.bbox_hidden2output = self._decoder_mlp(n_embd, self.bbox_dim)
            if self.class_dim > 0:
                self.class_hidden2output = self._decoder_mlp(n_embd, self.class_dim)
            if self.objectness_dim > 0:
                self.objectness_hidden2output = self._decoder_mlp(n_embd, self.objectness_dim)
            if self.objfeat_dim > 0:
                self.objfeat_hidden2output = self._decoder_mlp(n_embd, self.objfeat_dim)
        else:
            self.hidden2output = self._decoder_mlp(n_embd, self.channels)
    
    def forward(self, x, time, context=None, context_cross=None): 
        # x: (B, N, C)
        if context_cross is not None:
            raise NotImplemented    # TODO

        # initial processing
        if self.seperate_all:
            x_bbox = self.bbox_embedf(x[:, :, 0:self.bbox_dim])

            if self.class_dim > 0:
                start_index = self.bbox_dim
                x_class = self.class_embedf(
                    x[:, :, start_index:start_index+self.class_dim]
                )
            else:
                x_class = 0
            
            if self.objectness_dim > 0:
                start_index = self.bbox_dim+self.class_dim
                x_object = self.objectness_embedf(
                    x[:, :, start_index:start_index+self.objectness_dim]
                )
            else:
                x_object = 0
            
            if self.objfeat_dim > 0:
                start_index = self.bbox_dim+self.class_dim+self.objectness_dim
                x_objfeat = self.objfeat_embedf(
                    x[:, :, start_index:start_index+self.objfeat_dim]
                )
            else:
                x_objfeat = 0
                
            x = x_bbox + x_class + x_object + x_objfeat
        else:
            x = self.init_mlp(x)

        # transformer
        for block_idx in range(len(self.tf_blocks)):   
            x = self.tf_blocks[block_idx](
                x, time, context
            )

        # final processing
        if self.seperate_all:
            out  = self.bbox_hidden2output(x)
            if self.class_dim > 0:
                out_class = self.class_hidden2output(x)
                out = torch.cat([out, out_class], dim=2).contiguous()
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=2).contiguous()
            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=2).contiguous()
        else:
            out = self.hidden2output(x)
        
        return out