import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
# from thop import profile
# from transformer import 
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from MambaCD.changedetection.models.MDP import Mamba_Decoder_Pyramid
# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
# from Mamba_decoder import ChangeDecoder
import torchvision.models as models
import torch
import torch.nn as nn
import timm
import torchvision.models as models
# Load MiT-B0 model (SegFormer backbone)
class SwinFPN(nn.Module):
    def __init__(self,dim):
        super(SwinFPN, self).__init__()
        # Load pre-trained MiT-B0 from timm
        # print(timm.models.create_model('swin_base_patch4_window7_224').default_cfg)
        # self.backbone = timm.create_model('mit_b0', pretrained=True, features_only=True)
        # 加载预训练的 Segformer 模型 (基于 MIT-B0)
        self.backbone = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        # self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        # self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True,num_classes=10, features_only=True,global_pool='')
        self.dims=dim
        self.channel_first=False
    def forward(self, x):
        # Extract feature maps at different pyramid levels
        # Feature maps will be at 4 different stages as per the MiT-B0 architecture
        features = self.backbone(x)
        return features  # List of feature maps from 4 stages
class ResNetFPN(nn.Module):
    def __init__(self, backbone,dim):
        super(ResNetFPN,self).__init__()
        self.dims = dim
        self.channel_first = False
        # Extract layers from the pre-trained ResNet model
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
    def forward(self, x):
        # Feature pyramid levels from ResNet
        # Stage 0: output from conv1
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)  # Downsample after first block
        
        # Stage 1: output from layer1
        c2 = self.layer1(c1)
                # Stage 2: output from layer2
        c3 = self.layer2(c2)
        
        # Stage 3: output from layer3
        c4 = self.layer3(c3)
        
        # Stage 4: output from layer4
        c5 = self.layer4(c4)
        # import pdb
        # pdb.set_trace()
        # Return the feature maps (c2 to c5 are usually used for FPN)
        return [ c2, c3, c4, c5]

# Load a pre-trained ResNet model
resnet = models.resnet101(pretrained=True)
# resnet = models.resnet50(pretrained=True)
# resnet = models.resnet18(pretrained=True)
# resnet = models.resnet34(pretrained=True)
# Create the ResNet-FPN model
# fpn_model = ResNetFPN(resnet)

# # # Example input tensor
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size

# # Extract feature pyramids
# features = fpn_model(input_tensor)

# # Print the shapes of the feature maps
# for i, f in enumerate(features):
#     print(f"Feature map {i+1} shape: {f.shape}")
class MambaPyramid(nn.Module):
    def __init__(self, pretrained,**kwargs):
        super(MambaPyramid, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        # self.out_ch = out_ch
        # self.encoder=ResNetFPN(resnet,dim=[256,512,1024,2048])
        # self.encoder=ResNetFPN(resnet,dim=[64,128,256,512])
        # self.encoder = SwinFPN(dim=[256,512,1024,2048])
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 
        self.depth = kwargs['decoder_depths']
        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = Mamba_Decoder_Pyramid(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        self.ds = nn.ModuleList([])
        for i in range(self.depth-1):
            self.ds.append(nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1))
        # self.ds1 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        # self.ds2 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        # self.ds3 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        feature = []
        for index in range(len(pre_features)):
            feature.append(torch.cat([pre_features[index],post_features[index]],dim=1))
        output,output_ds = self.decoder(feature)
        output = self.main_clf(output)
        for i in range(self.depth-1):
            output_ds[i] = self.ds[i](output_ds[i])
        # output_ds[0] = self.ds1(output_ds[0])
        # output_ds[1] = self.ds2(output_ds[1])
        # output_ds[2] = self.ds3(output_ds[2])
      
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear',align_corners=False)
        for f in range(len(output_ds)):
            output_ds[f] = F.interpolate(output_ds[f], size=pre_data.size()[-2:], mode='bilinear',align_corners=False) 
        return output,output_ds
    

