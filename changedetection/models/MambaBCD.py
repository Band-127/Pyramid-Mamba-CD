import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
# from thop import profile
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from MambaCD.changedetection.models.ChangeDecoder import ChangeDecoder
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
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
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
# Create the ResNet-FPN model
# fpn_model = ResNetFPN(resnet)

# # # Example input tensor
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size

# # Extract feature pyramids
# features = fpn_model(input_tensor)

# # Print the shapes of the feature maps
# for i, f in enumerate(features):
#     print(f"Feature map {i+1} shape: {f.shape}")
class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
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
 

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        # self.decoder11 = ChangeDecoder(
        #     encoder_dims=self.encoder.dims,
        #     channel_first=self.encoder.channel_first,
        #     norm_layer=norm_layer,
        #     ssm_act_layer=ssm_act_layer,
        #     mlp_act_layer=mlp_act_layer,
        #     use_3x3=True,
        #     **clean_kwargs
        # )
        # print(self.encoder.dims)
        # self.drop_rate = kwargs['drop_rate'],
        # print(kwargs['drop_rate'])
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            # drop_rate = drop_rate,
            **clean_kwargs
        )
        # self.decoder2 = ChangeDecoder(
        #     encoder_dims=self.encoder.dims,
        #     channel_first=self.encoder.channel_first,
        #     norm_layer=norm_layer,
        #     ssm_act_layer=ssm_act_layer
        #     mlp_act_layer=mlp_act_layer,
        #     use_3x3=True,
        #     **clean_kwargs
        # )

        self.main_clf = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        self.ds1 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        self.ds2 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        self.ds3 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def forward(self, pre_data, post_data):
        # Encoder processing
        # print(pre_data.shape)
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

      # pre_features,post_features= channel_exchange(pre_features,post_features)
        # feature = torch.cat([pre_features,post_features],dim=2)
        feature = []
        for index in range(len(pre_features)):
            feature.append(torch.cat([pre_features[index],post_features[index]],dim=1))
        # output = self.decoder()
        # feature = self.encoder(feature)
        # output1,output2 = self.decoder(pre_features),self.decoder(post_features)
        # import pdb
        # pdb.set_trace()
        # output = self.decoder(feature,feature)
        # pre_features,post_features= channel_exchange(pre_features,post_features)
        # Decoder processing - passing encoder outputs to the decoder
        # output = torch.add(self.decoder(pre_features,post_features),output)
        # output = torch.cat([output1,output2],dim=1)
        output,output_ds = self.decoder(feature)
        output = self.main_clf(output)
        output_ds[0] = self.ds1(output_ds[0])
        output_ds[1] = self.ds2(output_ds[1])
        output_ds[2] = self.ds3(output_ds[2])
       
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear',align_corners=False)
        for f in range(len(output_ds)):
            output_ds[f] = F.interpolate(output_ds[f], size=pre_data.size()[-2:], mode='bilinear',align_corners=False)
        # import pdb
        # pdb.set_trace()
        return output,output_ds
    

def channel_exchange(pre,post):
    
    for index in range(len(pre)):
        bs,c,h,w=pre[index].shape
        interact_mat=generate_checkerboard_tensor(bs,c,h).to('cuda')
        # exchange_module=interact_mat.unsqueeze(0).unsqueeze(0).repeat(bs,c,1,1).to('cuda')
        # print()
        # import pdb
        # pdb.set_trace()
        # exchange_module=torch.zeros(bs,c,h,w).to('cuda')
        # exchange_module[:, : :2,:,:]=1
        pre_change=interact_mat[index]*pre[index]+(1-interact_mat[index])*post[index]
        post_change=interact_mat[index]*post[index]+(1-interact_mat[index])*pre[index]
        pre[index]=pre_change
        post[index]=post_change
        del pre_change
        del post_change
    return pre,post

def generate_checkerboard_tensor(b,c,h):  
    # 创建一个h*h的全0矩阵  
    tensor = torch.zeros(h, h, dtype=torch.int)  
      
    # 利用行索引和列索引的和的奇偶性来生成棋盘格模式  
    row_indices = torch.arange(h)  
    col_indices = torch.arange(h)  
    condition = (row_indices[:, None] + col_indices) % 2 == 1  
      
    # 使用torch.where根据条件赋值  
    tensor = torch.where(condition, torch.ones_like(tensor), tensor)  
    expanded_tensor = tensor.unsqueeze(0).unsqueeze(0).expand(b, c, h, h).clone()  
    # import pdb
    # pdb.set_trace()
    # 注意：这里使用了unsqueeze来添加两个新的维度，然后expand来“扩展”这些维度的大小（但实际上只是复制）  
    # clone()是可选的，但如果你打算在之后修改expanded_tensor而不影响原始tensor，那么使用clone()是个好习惯  
      
    return expanded_tensor 
   
    # return tensor