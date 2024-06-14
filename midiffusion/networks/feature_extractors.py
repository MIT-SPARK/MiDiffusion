import torch
from torch import nn
from torchvision import models

from .frozen_batchnorm import FrozenBatchNorm2d


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks"""
    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, data):
        raise NotImplementedError


class ResNet18(BaseFeatureExtractor):
    """Build a feature extractor using the pretrained ResNet18 architecture for
    image based inputs"""
    """https://github.com/nv-tlabs/ATISS/blob/master/scene_synthesis/networks/feature_extractors.py"""
    def __init__(self, freeze_bn, input_channels, feature_size):
        super(ResNet18, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.resnet18(weights=None)
        if freeze_bn:
            FrozenBatchNorm2d.freeze(self._feature_extractor)

        self._feature_extractor.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, X):
        return self._feature_extractor(X)


class PointNet_Point(BaseFeatureExtractor):
    """The 'pointnet_simple' encoder in train script"""
    """https://github.com/QiuhongAnnaWei/LEGO-Net/blob/main/model/floorplan_encoder.py"""

    def __init__(self, activation=torch.nn.LeakyReLU(), nfpbpn=256, feat_units=[4, 64, 64, 512, 511]):
        """ feat_units[0] should be 4 (x, y, nx, ny for fpbpn) (in_dim)
            feat_units[-1] should be transformer_input_d-1 (out_dim)
            maxpool happens before the last linear layer
        """
        super().__init__()
        self.activation = activation
        self._feature_size = feat_units[-1]

        layers = []
        for i in range(1, len(feat_units)): 
            layers.append(torch.nn.Linear(feat_units[i-1], feat_units[i]))
        self.layers = torch.nn.ModuleList(layers)
        
        self.fp_maxpool= torch.nn.MaxPool1d(nfpbpn) # nfpbpn -> 1
        

    def forward(self, fpbpn):
        """ fpbpn: [batch_size, nfpbp=250, 4], floor_plan_boundary_points_normals """

        B = fpbpn.shape[0]

        for i in range(len(self.layers)-1): # first 3 layers
            fpbpn = self.activation(self.layers[i](fpbpn))      # [B, nfpbp, feat_units[-2]]
        
        fpbpn = fpbpn.permute((0,2,1))                          # [B, feat_units[-2], nfpbp]
        scene_fp_feat = self.fp_maxpool(fpbpn).reshape(B, -1)   # [B, feat_units[-2], 1] -> [B, feat_units[-2]]

        return self.layers[-1](scene_fp_feat)                   # [B, feat_units[-1]]


class PointNet_Line(nn.Module):
    """ The 'pointnet' encoder in train script. 
        Has 3 sections: point processing, line processing, floor plan feature processing """
    def __init__(self, activation=torch.nn.LeakyReLU(), maxnfpoc=25, corner_feat_units = [2, 64, 128], line_feat_units = [256, 512, 1024], fp_units = [1024, 511]):
        """ corner_feat_units[0] should be pos_dim (in_dim)
            line_feat_units[0] should be corner_feat_units[-1]*2
            fp_units[0] should be line_feat_units[-1]
            fp_units[-1] should be transformer_input_d-1 (out_dim)
        """
        super().__init__()
        self.activation = activation # torch.nn.LeakyReLU()
        self._feature_size = fp_units[-1]

        corner_feat = []
        for i in range(1, len(corner_feat_units)): # 2 layers 
            corner_feat.append(torch.nn.Linear(corner_feat_units[i-1], corner_feat_units[i]))
        self.corner_feat = torch.nn.ModuleList(corner_feat)
        
        line_feat = []
        for i in range(1, len(line_feat_units)): # 2 layers
            line_feat.append(torch.nn.Linear(line_feat_units[i-1], line_feat_units[i]))
        self.line_feat = torch.nn.ModuleList(line_feat)
        self.fp_maxpool= torch.nn.MaxPool1d(maxnfpoc) # maxnfpoc -> 1
        
        # want floor plan to appear as an object token
        fp_feat = []
        for i in range(1, len(fp_units)): # 1 layer
            fp_feat.append(torch.nn.Linear(fp_units[i-1], fp_units[i]))
        self.fp_feat = torch.nn.ModuleList(fp_feat)


    def forward(self, fpoc, nfpc):
        """ fpoc        : [batch_size, maxnfpoc, pos=2], with padded 0 beyond the num of floor plan ordered corners for each scene
            nfpc        : [batch_size], num of floor plan ordered corners for each scene
        """
        device = fpoc.device
        
        # each point -> MLP -> each point has point features
        for i in range(len(self.corner_feat)-1):
            fpoc = self.activation(self.corner_feat[i](fpoc))
        fpoc = self.corner_feat[-1](fpoc)  # [B, nfpv, corner_feat_units[-1]=128]

        # concatenate 2 point features for each line pair -> MLP -> maxpool
        B, maxnfpoc, cornerpt_feat_d = fpoc.shape[0], fpoc.shape[1], fpoc.shape[2] # maxnfpoc = maxnfpoc
        line_pairpt_input = torch.zeros(B, maxnfpoc, cornerpt_feat_d*2).to(device)
        for s_i in range(B):
            for l_i in range(nfpc[s_i]): # otherwise padded with 0; clockwise ordering of lines
                line_pairpt_input[s_i, l_i, :cornerpt_feat_d] = fpoc[s_i, l_i, :] # index slicing does not make copy
                line_pairpt_input[s_i, l_i, cornerpt_feat_d:] = fpoc[s_i, (l_i+1)%nfpc[s_i], :]
        line_pairpt_input = line_pairpt_input.to(device)

        for i in range(len(self.line_feat)-1):
            line_pairpt_input = self.activation(self.line_feat[i](line_pairpt_input))
        line_pairpt_input = self.line_feat[-1](line_pairpt_input)  # [B, nfpv, line_feat_units[-1]=1024]

        line_pairpt_input_padded = torch.zeros(line_pairpt_input.shape).to(device) # [B, nfpv, 1024]
        for s_i in range(B): 
            line_pairpt_input_padded[s_i, :nfpc[s_i], :] = line_pairpt_input[s_i, :nfpc[s_i], :]
            line_pairpt_input_padded[s_i, nfpc[s_i]:, :] = line_pairpt_input[s_i, 0, :] # duplicate first feat to not impact maxpool
        line_pairpt_input_padded = line_pairpt_input_padded.to(device)
        
        line_pairpt_input_padded = line_pairpt_input_padded.permute((0,2,1))  # [B, 1024, nfpv]
        scene_fpoc = self.fp_maxpool(line_pairpt_input_padded).reshape(B, -1) # [B, 1024, 1] -> [B, 1024]

        # One floor plan feat per scene -> MLP -> input to transformer as last obj token
        for i in range(len(self.fp_feat)-1):
            scene_fpoc = self.activation(self.fp_feat[i](scene_fpoc))
        scene_fpoc = torch.unsqueeze(self.fp_feat[-1](scene_fpoc), dim=1)  # [B, None->1, transformer_input_d-1]

        return scene_fpoc


def get_feature_extractor(name, **kwargs):
    """Based on the name return the appropriate feature extractor."""
    if name == "resnet18":
        return ResNet18(**kwargs)
    elif name == "pointnet_simple":
        return PointNet_Point(**kwargs)
    elif name == "pointnet":
        return PointNet_Line(**kwargs)
    else:
        raise NotImplemented
