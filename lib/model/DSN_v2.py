'''
This file is modified from:
https://github.com/deepmind/kinetics-i3d/i3d.py
'''

import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import cv2
import os, math
import sys
from .DTN import DTNNet
from .DTN_v2 import DTNNet as DTNNetV2
from .FRP import FRP_Module
from .utils import *

import os, math
import sys
sys.path.append('../../')
from collections import OrderedDict
from utils import load_pretrained_checkpoint
import logging

class RCMModule(nn.Module):
    def __init__(self, args, dim_head=16):
        super(RCMModule, self).__init__()
        args.recoupling = False
        self.args = args
        self._distill = True
        self.heads = args.sample_duration
        self.inp_dim = args.sample_duration
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)

        # Self Attention Layers
        self.q = nn.Linear(self.inp_dim, dim_head * self.heads, bias=False)
        self.k = nn.Linear(self.inp_dim, dim_head * self.heads, bias=False)
        self.scale = dim_head ** -0.5

        # Distill MLP
        if self._distill:
            self.TM_project = nn.Sequential(
                nn.Linear(self.inp_dim, self.inp_dim*2, bias=False),
                nn.GELU(),
                nn.Linear(self.inp_dim*2, self.inp_dim, bias=False),
                nn.LayerNorm(self.inp_dim),
            )
        temp_out = args.sample_duration//2 if args.sample_duration == 64 else args.sample_duration
        self.linear = nn.Linear(self.inp_dim,  512)

    def forward(self, x):
        b, c, t, h, w = x.shape
        residual = x

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.avg_pool(x)
        x = rearrange(x, '(b t) c h w -> b c (t h w)', t=t)

        # x = self.norm(x)
        q, k = self.q(x), self.k(x)
        v = rearrange(residual, 'b c t h w -> b t c (h w)')

        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # attn = attn.mean(-2, keepdim=True).transpose(2,3)
        # out = v * attn.expand_as(v)

        out = rearrange(out, 'b t c (h w) -> b c t h w', h=h, w=w)
        # out += residual

        if self._distill:
            temporal_embedding = self.avg_pool3d(out.permute(0, 2, 1, 3, 4)).squeeze()
            temporal_project = self.TM_project(temporal_embedding)
            temporal_weight = torch.sigmoid(temporal_project)[:, None, :, None, None]
            out = out * temporal_weight.expand_as(out)
            temporal_weight = temporal_weight.squeeze()

        out += residual
        return out, self.linear(temporal_project)

class SMSBlock(nn.Module):
    def __init__(self, channel_list, kernel_size=None, stride=None, padding=0, name='i3d'):
        super(SMSBlock, self).__init__()

        in_channels, hidden_channels, out_channels =  channel_list
        self.end_points = {}

        end_point = 'Mixed1'
        self.end_points[end_point] = SpatialInceptionModule(in_channels, hidden_channels, name + end_point)

        end_point = 'Mixed2'
        self.end_points[end_point] = SpatialInceptionModule(sum([hidden_channels[0], hidden_channels[2], 
                                                                hidden_channels[4], hidden_channels[5]]), out_channels, name + end_point)
        if kernel_size is not None:
            end_point = 'MaxPool3d_3x3'
            self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=kernel_size, stride=stride,
                                                            padding=padding)
        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.end_points.keys():
            x = self._modules[end_point](x)
        return x

class Channel_Pooling(nn.Module):
    def __init__(self):
        super(Channel_Pooling, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1), padding=0)
    
    def forward(self, x):
        # x size: torch.Size([16, 512, 16, 7, 7])
        x = x.transpose(1, 2)
        x = self.max_pool(x)
        return x.transpose(1, 2)
class DSNNetV2(nn.Module):
    def __init__(self, args, num_classes=400, spatial_squeeze=True, name='inception_i3d', in_channels=3, 
                 pretrained: bool = False,
                 sms_depth: int = 3,
                 dropout_spatial_prob: float=0.0,
                 frames_drop_rate: float=0.0):

        super(DSNNetV2, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self.args = args

        self.stem = nn.Sequential(
            Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[1, 7, 7],
                                            stride=(1, 2, 2), padding=(0, 3, 3), name=name + 'Conv3d_1a_7x7'),
            MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0),
            Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + 'Conv3d_2b_1x1'),
            Unit3D(in_channels=64, output_channels=192, kernel_shape=[1, 3, 3],
                                            padding=(0, 1, 1), name=name + 'Conv3d_2c_3x3'),
            MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        )

        '''
        Spatial Multi-scale Features Learning
        '''
        sms_block = [
            # input_dim, hidden_dim, output_dim
            [192, [64, 96, 128, 16, 32, 32], [128, 128, 192, 32, 96, 64]],
            [128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], [160, 112, 224, 24, 64, 64]],
            [160 + 224 + 64 + 64, [256, 160, 320, 32, 128, 128], [384, 192, 384, 48, 128, 128]],
        ]
        assert len(sms_block) == sms_depth

        self.SMS_layers = nn.ModuleList([])
        for i in range(sms_depth):
            if i == 0:
                self.SMS_layers.append(
                    SMSBlock(sms_block[i], kernel_size=[1,3,3], stride=(1,2,2), padding=0)
                )
            elif i==1:
                self.SMS_layers.append(
                    SMSBlock(sms_block[i], kernel_size=[1,2,2], stride=(1,2,2), padding=0)
                )
            elif i==2:
                self.SMS_layers.append(SMSBlock(sms_block[i], kernel_size=[1,1,1], stride=(1,1,1), padding=0))
                self.SMS_layers.append(Channel_Pooling()),
            self.SMS_layers.append(RCMModule(args))
        self.LinearMap = nn.Sequential(
            nn.Dropout(dropout_spatial_prob),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        self.dtn = DTNNetV2(args, num_classes=self._num_classes)
        self.rrange = Rearrange('b c t h w -> b t c h w')
        self.frames_droupout = torch.nn.Dropout2d(p=frames_drop_rate, inplace=False)

        # Feature visualization
        self.feat = None
        self.visweight = None
    def get_visualization(self):
        return self.feat, self.visweight

    def build(self):
        for k in self.SMS_layers.keys():
            self.add_module(k, self.SMS_layers[k])

    def forward(self, x, garr=None):
        inp = x
        x = self.stem(x)
        temp_out = []
        for i, sms_layer in enumerate(self.SMS_layers):
            x = sms_layer(x)
            if isinstance(x, tuple):
                x, temp_w = x
                temp_out.append(temp_w)
            if i == 1:
                f = x
        self.feat = x.data
        x = self.avg_pool(x).view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x = self.LinearMap(x)
        x = self.frames_droupout(x)

        cnn_vison = self.rrange(f.sum(dim=1, keepdim=True))
        self.visweight = torch.sigmoid(x[0])
        # logits, _, (att_map, cosin_similar, MHAS, visweight) = self.dtn(x)
        x, (xs, xm, xl) = self.dtn(x)

        target_out = []
        for j in range(len(self.dtn.multi_scale_transformers)):
            target_out.append(self.dtn.multi_scale_transformers[j][2].get_classEmbd())

        return (x, xs, xm, xl), (temp_out, target_out)


