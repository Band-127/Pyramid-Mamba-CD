import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import einops


class Mamba_Decoder_Pyramid(nn.Module):
    def __init__(self, encoder_dims,channel_first, norm_layer, ssm_act_layer, mlp_act_layer, use_3x3, drop_rate=0.0,**kwargs):
        super(Mamba_Decoder_Pyramid, self).__init__()
        
        self.embeding_dim=[2*encoder_dims[i] for i in range(len(encoder_dims))]
        encoder_dims = [encoder_dims[i]*2 for i in range(len(encoder_dims))]
        self.shape=[64,32,16,8]
        # 96 192 384 768
        self.conv_list = nn.ModuleList([])
        for i in range(1,len(self.embeding_dim)):
            if not use_3x3:
                self.conv_list.append(nn.Sequential(nn.Conv2d(kernel_size=1, in_channels = encoder_dims[i], out_channels = 256),
                                                nn.GroupNorm(32,256)))
            else:
                if i == len(self.embeding_dim)-1:
                    self.conv_list.append(nn.Sequential(nn.Conv2d(kernel_size=1, in_channels = encoder_dims[i], out_channels = 256),
                                                nn.GroupNorm(32,256)))
                    self.conv_list.append(nn.Sequential(nn.Conv2d(encoder_dims[i], 256, kernel_size=3, stride=2, padding=1),  # 3x3conv s=2 -> 256channel
                    nn.GroupNorm(32, 256)))
                else:
                    self.conv_list.append(nn.Sequential(nn.Conv2d(kernel_size=1, in_channels = encoder_dims[i], out_channels = 256),
                                              nn.GroupNorm(32,256)))
        self.lvl_embed = nn.Parameter(torch.rand(4,1,256))
        # 4 192
        self.depth = kwargs['decoder_depths']
        self.drop_rate = drop_rate
        self.model_list=[]
        for _ in range(self.depth):
            self.model_list.append(nn.Sequential(
                nn.Conv2d(kernel_size=1, in_channels=256, out_channels=256),
                Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                VSSBlock(hidden_dim=256, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),Permute(0, 3, 1, 2) if not channel_first else nn.Identity()))
            

        self.vss = nn.ModuleList(self.model_list)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.concat_layer=upsample_block()
        self.smooth_layer = nn.ModuleList([ResBlock(in_channels=256, out_channels=256, drop_rate = self.drop_rate,stride=1) for _ in range(self.depth+1)])
        self.output_perform = nn.Conv2d(kernel_size=1,in_channels= 4 * 256,out_channels = 256)
        self.apply(self.parameter_init)
        self.ds = nn.ModuleList([])
        for _ in range(self.depth - 1):
            self.ds.append(nn.Conv2d(kernel_size=1,in_channels= 4 * 256,out_channels = 256))

    def align_and_cat(self,pre_level_f):
        bs,c,w,h=pre_level_f.shape
        pre_level=pre_level_f.reshape(bs,c//2,w,h*2)
        return pre_level
    
    def channel_mapper_and_cat(self,feature):
        bs,c, __ , __ = feature[0].shape
        c = 256
        # print(feature[0].shape)
        output=[]
        # feature[0]=feature[0].reshape(bs,c,-1)
        length=len(feature) #4
        # import pdb
        # pdb.set_trace()
        for i in range(1,length):
            if i==length-1:
                output.append((self.conv_list[i-1](feature[i])).reshape(bs,c,-1))
                output.append((self.conv_list[i](feature[i])).reshape(bs,c,-1))
                # feature[i]=(self.conv_list[i](feature[i])).reshape(bs,c,-1)
                # feature_tmp=(self.conv_list[i](feature[i])).reshape(bs,c,-1)
            else:
                output.append((self.conv_list[i-1](feature[i])).reshape(bs,c,-1))
        for lvl,encoding_f in enumerate(output):
            encoding_f += self.lvl_embed[lvl].view(1,-1,1)
        return torch.cat([output[i] for i in range(4)],dim=2)

    
    def batch_split_and_smooth(self,features,shape_list,index,name):
        bs,c,__, l = features.shape
        [feature_0,feature_1,feature_2,feature_3] = self.shuffle_seq_order(features)
        # import pdb
        # pdb.set_trace()
        feature_0 = self.smooth_layer[index](feature_0.reshape(bs,c,shape_list[0],-1))
        feature_1 = self.smooth_layer[index](feature_1.reshape(bs,c,shape_list[1],-1))
        feature_2 = self.smooth_layer[index](feature_2.reshape(bs,c,shape_list[2],-1))
        # import pdb
        # pdb.set_trace()
        feature_3 = self.smooth_layer[index](feature_3.reshape(bs,c,shape_list[3],-1))
        feature_mid = [feature_0,feature_1,feature_2,feature_3]
        # print(index)
        blk_index= index
        if name=='mid':
            feature = [feature_0,feature_1,feature_2,feature_3]
            for index in range(len(feature)):
                feature[index] = feature[index].reshape(bs,c,1,-1)
                # import pdb
                # pdb.set_trace()
            return torch.cat(feature,dim=-1),feature_mid
        else:
            return [feature_0,feature_1,feature_2,feature_3]
    
    def parameter_init(self,module):
        if isinstance(module, nn.Conv2d):
        # He 初始化 (Kaiming 初始化)
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, features):
        # features[0] = None
        bs,c,w,h  = features[1].shape
        # w1,w2,w3,w4 = w,w//2,w//4,w//8
        features_mapper = self.channel_mapper_and_cat(features)
        # feature_mapper = features_mapper
        del features
        # print(features.shape)
        # import pdb
        # pdb.set_trace()
        features_mapper = features_mapper.unsqueeze(-2)
        bs,c,_,l=features_mapper.shape
        assert l == w*w*(1+1/4+1/16+1/64)
        max_shape=int(w)
        self.shape_list = [max_shape,max_shape//2,max_shape//4,max_shape//8]
        feature_ds = []
        # pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = features_mapper
        for blk_index in range(len(self.vss)):

            # print(len(self.vss))
            features_mapper,feature_dsup = self.batch_split_and_smooth(self.vss[blk_index](features_mapper),shape_list=self.shape_list,name='mid',index=blk_index)
            if blk_index!=len(self.vss)-1:
                feature_dsup =  self.upsample(self.ds[blk_index](self.concat_layer(feature_dsup)))
                # import pdb
                # pdb.set_trace()
                feature_ds.append(feature_dsup)
            
        feature_mapper = self.batch_split_and_smooth(features_mapper,shape_list=self.shape_list,name='last',index=blk_index+1)
        # import pdb
       # pdb.set_trace()
        feature_mapper = self.concat_layer(feature_mapper)
       # feature_concat = self.upsample(feature_concat)
        return self.upsample(self.smooth_layer[-1](self.output_perform(feature_mapper))),feature_ds
    def shuffle_seq_order(self,features):
        '''
        a way to shuffle the sequences, realize the different orders.
        seq tensor[N,L,C]
        shape_list: save the shape to get the size of tensor.
        reverse after a block
        '''
        # self.shape_list.reverse()
        
        shuffle_seq = []
        start = 0
        for i in range(4):
            shuffle_seq.append(features[:,:,:,start:start+self.shape_list[i]**2])
            start+=self.shape_list[i]**2
        # feature_0,feature_1,feature_2,feature_3= features[:,:,:,:shape_list[0]],features[:,:,:,shape_list[0]:shape_list[0]+shape_list[1]],features[:,:,:,shape_list[0]+shape_list[1]:+shape_list[2]+shape_list[0]+shape_list[1]],features[:,:,:,shape_list[2]+shape_list[0]+shape_list[1]:]
        return shuffle_seq

class upsample_block(nn.Module):
    def __init__(self):
        super(upsample_block,self).__init__()
        self.upsample_factor = [1, 2, 4, 8]
        # self.upsample_block=nn.ModuleList()
        upsample_list = []
        for factor in self.upsample_factor:
            upsample_list.append(nn.Upsample(scale_factor=factor,mode='bilinear'))
        self.upsample_block=nn.ModuleList(upsample_list)
    def forward(self,feature):
        # for i in range(4):
        bs,c,w1,_=feature[0].shape
        w = max([feature[i].shape[-2] for i in range(4)])
        feature_shape = [w,w//2,w//4,w//8]
        if w==w1:
            for index in range(0,len(feature)):
                # import pdb
                # pdb.set_trace()
                feature[index] = feature[index].reshape(bs,c,feature_shape[index],-1)
            # if index >=1:
                feature[index] = self.upsample_block[index](feature[index])
            # print(feature[index].shape)
        else:
            for index in range(0,len(feature)):
                feature[index] = feature[index].reshape(bs,c,feature_shape[len(feature)-index-1],-1)
                feature[index] = self.upsample_block[len(feature)-1-index](feature[index])
        return torch.cat(feature,dim=1)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(self.conv2(out))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
