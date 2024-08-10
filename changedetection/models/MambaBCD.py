import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
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

class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
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
        self.decoder1 = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            **clean_kwargs
        )
        self.decoder2 = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=384, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        # print(pre_data.shape)
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        # feature = torch.cat([pre_data,post_data],dim=2)
        # feature = self.encoder(feature)
        output1,output2 = self.decoder1(pre_features),self.decoder2(post_features)
        # import pdb
        # pdb.set_trace()
        # output = self.decoder(feature,feature)
        # pre_features,post_features= channel_exchange(pre_features,post_features)
        # Decoder processing - passing encoder outputs to the decoder
        # output = torch.add(self.decoder(pre_features,post_features),output)
        output = torch.cat([output1,output2],dim=1)
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output
    

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
        pre_change=interact_mat*pre[index]+(1-interact_mat)*post[index]
        post_change=interact_mat*post[index]+(1-interact_mat)*pre[index]
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