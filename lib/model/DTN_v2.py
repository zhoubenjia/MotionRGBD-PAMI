'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from torch.autograd import Variable
from torch import nn, einsum
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, helpers, DropPath

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import random, math
from .utils import *
from .trans_module import *
from utils import uniform_sampling

import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
import cv2

np.random.seed(123)
random.seed(123)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., apply_transform=False, knn_attention=0.7):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.cls_embed = [None for _ in range(depth)]
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                       apply_transform=apply_transform, knn_attention=knn_attention)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for ii, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            self.cls_embed[ii] = x[:, 0]
        return x

    def get_classEmbd(self):
        return self.cls_embed

class clsToken(nn.Module):
    def __init__(self, frame_rate, inp_dim):
        super().__init__()
        self.frame_rate = frame_rate
        num_patches = frame_rate
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, inp_dim))
        
    def forward(self, x):
        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, :(N + 1)]
        return x

class DTNNet(nn.Module):
    def __init__(self, args, num_classes=249, inp_dim=512, dim_head=64, hidden_dim=768,
                heads=8, pool='cls', dropout=0.1, emb_dropout=0.1, mlp_dropout=0.0, branch_merge='pool',
                 init: bool = False,
                 warmup_temp_epochs: int = 30,
                 branchs=3,
                 dynamic_tms=True):
        super().__init__()

        self._args = args

        print('Temporal Resolution:' )
        frame_rate = args.sample_duration // args.intar_fatcer
        # names = self.__dict__
        self.cls_tokens = nn.ModuleList([])
        dynamic_kernel = []
        for i in range(branchs):
            # names['cls_token_' + str(i)] = nn.Parameter(torch.randn(1, 1, frame_rate))
            self.cls_tokens.append(clsToken(frame_rate, inp_dim))
            print(frame_rate)
            dynamic_kernel.append(int(frame_rate**0.5))
            frame_rate += args.sample_duration // args.intar_fatcer

        '''
        constract multi-branch structures
        '''
        trans_depth = args.N
        self.multi_scale_transformers = nn.ModuleList([])
        for ii in range(branchs):
            self.multi_scale_transformers.append(
                nn.ModuleList([
                    TemporalInceptionModule(inp_dim, [160,112,224,24,64,64], kernel_size=dynamic_kernel[ii] if dynamic_tms else 3),
                    MaxPool3dSamePadding(kernel_size=[3, 1, 1], stride=(1, 1, 1), padding=0),
                    Transformer(inp_dim, trans_depth, heads, dim_head, mlp_dim=hidden_dim, dropout=emb_dropout, knn_attention=args.knn_attention),
                    nn.Sequential(
                        nn.LayerNorm(inp_dim),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(inp_dim, num_classes))
                ]))

        # num_patches = args.sample_duration
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, inp_dim))

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.branch_merge = branch_merge
        warmup_temp, temp = map(float, args.temp)
        self.temp_schedule = np.concatenate((
            np.linspace(warmup_temp,
                        temp, warmup_temp_epochs),
            np.ones(args.epochs - warmup_temp_epochs) * temp
        ))
        # self.show_res = Rearrange('b t (c p1 p2) -> b t c p1 p2', p1=int(small_dim ** 0.5), p2=int(small_dim ** 0.5))
  
        if init:
            self.init_weights()

    def TC_forward(self):
        return self.tc_feat

    # @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x):  # x size: [2, 64, 512]
        B, N, C = x.shape

        # Add position embedding
        # x += self.pos_embedding

        # Local-Global features capturing
        outputs = []
        temp = self.temp_schedule[self._args.epoch]
        for cls_token, (TCNN, MaxPool, TransBlock, mlp) in zip(self.cls_tokens, self.multi_scale_transformers):
            # cls_token = self.__dict__['cls_token_{}'.format(i)]

            sl = uniform_sampling(x.size(1), cls_token.frame_rate, random=self.training)
            sub_x = x[:, sl, :]
            sub_x = sub_x.permute(0, 2, 1).view(B, C, -1, 1, 1)
            sub_x = MaxPool(TCNN(sub_x))
            sub_x = sub_x.permute(0, 2, 1, 3, 4).view(B, -1, C)
            
            sub_x = cls_token(sub_x)
            sub_x = TransBlock(sub_x)
            sub_x = sub_x[:, 0, :]

            out = mlp(sub_x)
            outputs.append(out / temp)

        # Multi-branch fusion
        if self.branch_merge == 'sum':
            x = torch.zeros_like(out)
            for out in outputs:
                x += out
        elif self.branch_merge == 'pool':
            x = torch.cat([out.unsqueeze(-1) for out in outputs], dim=-1)
            x = self.max_pool(x).squeeze()
        return x, outputs