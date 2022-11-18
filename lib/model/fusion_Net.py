'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.autograd import Variable
from collections import OrderedDict

import numpy as np

import os
import sys
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

sys.path.append(['../../', '../'])
from utils import load_pretrained_checkpoint, load_checkpoint, SoftTargetCrossEntropy, concat_all_gather, uniform_sampling
import logging
# from .DSN_Fusion import DSNNet
from .DSN_v2 import DSNNetV2
from .DTN_v2 import DTNNet as DTNNetV2
from .DTN_v2 import Transformer, clsToken
from .trans_module import *

class Encoder(nn.Module):
    def __init__(self, C_in, C_out, dilation=2):
        super(Encoder, self).__init__()
        self.enconv = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_in, C_in // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in // 2),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_in // 2, C_in // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in // 4),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_in // 4, C_out, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x1, x2):
        b, c = x1.shape
        x = torch.cat((x1, x2), dim=1).view(b, -1, 1, 1)
        x = self.enconv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, C_in, C_out, dilation=2):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(C_in, C_out // 4, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 4),
            nn.ReLU(),

            nn.Conv2d(C_out // 4, C_out // 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x


class FusionModule(nn.Module):
    def __init__(self, channel_in=1024, channel_out=256, num_classes=60):
        super(FusionModule, self).__init__()
        self.encoder = Encoder(channel_in, channel_out)
        self.decoder = Decoder(channel_out, channel_in)
        self.efc = nn.Conv2d(channel_out, num_classes, kernel_size=1, padding=0, bias=False)

    def forward(self, r, d):
        en_x = self.encoder(r, d)  # [4, 256, 1, 1]
        de_x = self.decoder(en_x)
        en_x = self.efc(en_x)
        return en_x.squeeze(), de_x

class DTN(DTNNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        B, N, C = x.shape

        # Local-Global features capturing
        outputs, tem_feat = [], []
        temp = self.temp_schedule[self._args.epoch]
        for cls_token, (TCNN, MaxPool, TransBlock, mlp) in zip(self.cls_tokens, self.multi_scale_transformers):

            sl =uniform_sampling(x.size(1), cls_token.frame_rate, random=self.training)
            sub_x = x[:, sl, :]
            sub_x = sub_x.permute(0, 2, 1).view(B, C, -1, 1, 1)
            sub_x = MaxPool(TCNN(sub_x))
            sub_x = sub_x.permute(0, 2, 1, 3, 4).view(B, -1, C)
            
            sub_x = cls_token(sub_x)
            sub_x = TransBlock(sub_x)
            sub_x = sub_x[:, 0, :]
            tem_feat.append(sub_x.unsqueeze(-1))

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

        return x, outputs, torch.cat(tem_feat, dim=-1).mean(-1)

class DSN(DSNNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtn = DTN(self.args, num_classes=self._num_classes)

    def forward(self, x, endpoint=None):
        # if endpoint=='spatial':
        x = self.stem(x)
        temp_out = []
        for i, sms_layer in enumerate(self.SMS_layers):
            x = sms_layer(x)
            if isinstance(x, tuple):
                x, temp_w = x
                temp_out.append(temp_w)
        self.feat = x
        x = self.avg_pool(x).view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x = self.LinearMap(x)
        x = self.frames_droupout(x)

        spati_feat = x

        self.visweight = torch.sigmoid(x[0])
        target_out = []
        for j in range(len(self.dtn.multi_scale_transformers)):
            target_out.append(self.dtn.multi_scale_transformers[j][2].get_classEmbd())
        # return x, (temp_out, target_out)
        
        x, (xs, xm, xl), tem_feat = self.dtn(x)
        return (x, xs, xm, xl), spati_feat, tem_feat, (temp_out, target_out)

class AttentionNet(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64, mlp_dim=768, dropout=0.1, knn_attention=True, topk=0.7):
        super(AttentionNet, self).__init__()
        self.knn_attention = knn_attention
        self.topk = topk
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.k = nn.Linear(dim, inner_dim, bias=False)
        self.v = nn.Linear(dim, inner_dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        # self.map = nn.Linear(inner_dim, dim, bias=True)
    
    def forward(self, x_r, x_d):
        b, n, c, h = *x_r.shape, self.heads

        q, k, v = self.q(x_r), self.k(x_d), self.v(x_r)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.knn_attention:
            mask = torch.zeros(b, self.heads, n, n, device=x_r.device, requires_grad=False)
            index = torch.topk(dots, k=int(dots.size(-1)*self.topk), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            dots = torch.where(mask > 0, dots, torch.full_like(dots, float('-inf')))

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out) + x_r
        out = self.ffn(out) + out
        return out

class EnhanceModule(nn.Module):
    def __init__(self, dim=512):
        super(EnhanceModule, self).__init__()
        self.mlp_rgb = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.mlp_depth = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim*2)
    def forward(self, xr, xd):
        joint_feature = self.norm(torch.cat((xr, xd), dim=-1))
        score_grb = self.mlp_rgb(joint_feature)
        score_depth = self.mlp_depth(joint_feature)
        xr = xr * score_grb
        xd = xd * score_depth

        return xr, xd

class ComplementSpatial(nn.Module):
    def __init__(self, depths=2, dim=512):
        super(ComplementSpatial, self).__init__()

        self.att_nets = nn.ModuleList([])
        for _ in range(depths):
            self.att_nets.append(nn.ModuleList([
                EnhanceModule(dim),
                AttentionNet(dim),
                AttentionNet(dim)
            ]))
        self.norm = nn.LayerNorm(dim*2)
    
    def forward(self, xr, xd):
        b, n, c = xr.shape
        xr, xd = torch.split(self.norm(torch.cat((xr, xd), dim=-1)), [c, c], dim=-1)
        for EM, ANM, ANK in self.att_nets:
            xr, xd = EM(xr, xd)
            # xr, xd = ANM(xr, xd), ANK(xd, xr)
            cm = ANM(xr, xd)
            ck = ANK(xd, xr)
            xr, xd = cm, ck
        
        return xr, xd

class ComplementTemporal(nn.Module):
    def __init__(self, depths=2, dim=512):
        super(ComplementTemporal, self).__init__()

        self.att_nets = nn.ModuleList([])
        for _ in range(depths):
            self.att_nets.append(nn.ModuleList([
                # EnhanceModule(dim),
                AttentionNet(dim),
                AttentionNet(dim)
            ]))
        self.norm = nn.LayerNorm(dim*2)
    def forward(self, xr, xd):
        b, n, c = xr.shape
        xr, xd = torch.split(self.norm(torch.cat((xr, xd), dim=-1)), [c, c], dim=-1)
        for ANM, ANK in self.att_nets:
            # xr, xd = ANM(xr, xd), ANK(xd, xr)
            # xr, xd = EM(xr, xd)
            cm = ANM(xr, xd)
            ck = ANK(xd, xr)
            xr, xd = cm, ck
        return xr, xd

class SFNNet(nn.Module):
    def __init__(self, args, num_classes, pretrained, spatial_interact=False, temporal_interact=False):
        super(SFNNet, self).__init__()
        self.linear = nn.Linear(2, num_classes)
    def forward(self, logitr, logitd):
        b, c = logitr.shape
        softmaxr = torch.softmax(logitr, dim=-1)
        softmaxd = torch.softmax(logitd, dim=-1)
        cat_softmax = torch.cat((softmaxr.unsqueeze(-1), softmaxd.unsqueeze(-1)), dim=-1)
        output = self.linear(cat_softmax)
        output *= torch.eye(c, c, device=logitr.device, requires_grad=False)
        return output.sum(-1)


class CrossFusionNet(nn.Module):
    def __init__(self, args, num_classes, pretrained, spatial_interact=False, temporal_interact=False):
        super(CrossFusionNet, self).__init__()
        self._MES = torch.nn.MSELoss()
        self._BCE = torch.nn.BCELoss()
        self._CE = SoftTargetCrossEntropy()
        self.spatial_interact = spatial_interact
        self.temporal_interact = temporal_interact
        self.args = args
        self.frame_rate = args.sample_duration #//2 if args.sample_duration > 32 else args.sample_duration

        self.visweight = None
        self.feat = None
        self.pca_data = None
        self.target_data = None
        
        self.SCC_Module = ComplementSpatial(depths=args.scc_depth)
        self.temp_enhance_module = EnhanceModule(dim=512)
        self.TimesFormer = ComplementTemporal(depths=args.tcc_depth)

        # self.timesform1 = Transformer(dim=512, depth=2, heads=8, dim_head=64, mlp_dim=768, 
        #                             dropout=0.1)
        # self.cls_token1 = clsToken(self.frame_rate+1, 512)
        self.pos_embedding_M = nn.Parameter(torch.randn(1, self.frame_rate + 1, 512))

        # self.timesform2 = Transformer(dim=512, depth=2, heads=8, dim_head=64, mlp_dim=768, 
        #                             dropout=0.1)
        # self.cls_token2 = clsToken(self.frame_rate+1, 512)
        self.pos_embedding_K = nn.Parameter(torch.randn(1, self.frame_rate + 1, 512))

        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.max_pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

        if pretrained:
            load_pretrained_checkpoint(self, pretrained)
            logging.info("Load Pre-trained model state_dict Done !")

    def forward(self, hidden_feature):
        spatial_M, spatial_K, temporal_M, temporal_K = hidden_feature
        comple_features_M, comple_features_K = self.SCC_Module(spatial_M, spatial_K)
        b, n, c = comple_features_M.shape
        # if self.frame_rate > 32:
        #     comple_features_M = self.max_pool(comple_features_M.view(b, c, n, 1, 1)).view(b, n//2, c)
        #     comple_features_K = self.max_pool(comple_features_K.view(b, c, n, 1, 1)).view(b, n//2, c)

        temporal_enhance_M, temporal_enhance_K = self.temp_enhance_module(temporal_M, temporal_K)
        # temporal_enhance_M, temporal_enhance_K = temporal_M, temporal_K

        temporal_feature_M = self.norm1(torch.cat((temporal_enhance_M.unsqueeze(1), comple_features_M), dim=1))
        temporal_feature_M += self.pos_embedding_M
        # temporal_feature_M = self.cls_token1(temporal_feature_M)
        # temporal_feature_M = self.timesform1(temporal_feature_M)

        temporal_feature_K = self.norm2(torch.cat((temporal_enhance_K.unsqueeze(1), comple_features_K), dim=1))
        temporal_feature_K += self.pos_embedding_K
        # temporal_feature_K = self.cls_token2(temporal_feature_K)
        # temporal_feature_K = self.timesform2(temporal_feature_K)

        temporal_feature_M, temporal_feature_K = self.TimesFormer(temporal_feature_M, temporal_feature_K)

        out_M = self.classifier1(temporal_feature_M[:, 0])
        out_K = self.classifier2(temporal_feature_K[:, 0])

        normal_func = lambda x: concat_all_gather(F.normalize(x, p = 2, dim=-1))
        b, _ = normal_func(temporal_M).shape
        self.pca_data = torch.cat((normal_func(temporal_M),normal_func(temporal_K), normal_func(temporal_feature_M[:, 0]), normal_func(temporal_feature_K[:, 0])))
        self.target_data = torch.cat((torch.ones(b), torch.ones(b)+1, torch.ones(b)+2, torch.ones(b)+3))

        return (out_M,out_K), (None, torch.cat((temporal_feature_M[:, 0].unsqueeze(-1), temporal_feature_K[:, 0].unsqueeze(-1)), dim=-1))
    
    def get_cluster_visualization(self):
        return self.pca_data, self.target_data
    def get_visualization(self):
        return self.feat, self.visweight


class FeatureCapter(nn.Module):
    def __init__(self, args, num_classes=249, pretrained=None):
        super(FeatureCapter, self).__init__()
        self.args = args
        assert args.rgb_checkpoint and args.depth_checkpoint
        self.Modalit_rgb = DSN(args, num_classes=num_classes)
        self.Modalit_depth = DSN(args, num_classes=num_classes)

        rgb_checkpoint = args.rgb_checkpoint[args.FusionNet]
        self.strat_epoch_r, best_acc = load_checkpoint(self.Modalit_rgb, rgb_checkpoint)
        print(f'Best acc RGB: {best_acc}')
        depth_checkpoint = args.depth_checkpoint[args.FusionNet]
        self.strat_epoch_d, best_acc = load_checkpoint(self.Modalit_depth, depth_checkpoint)
        print(f'Best acc depth: {best_acc}')

    def forward(self, rgb, depth):
        self.args.epoch = self.strat_epoch_r - 1
        (logit_M, M_xs, M_xm, M_xl), spatial_M, temporal_M, temp_out_M = self.Modalit_rgb(rgb, endpoint='spatial')
        self.args.epoch = self.strat_epoch_d - 1
        (logit_K, K_xs, K_xm, K_xl), spatial_K, temporal_K, temp_out_K = self.Modalit_depth(depth, endpoint='spatial')
        return (logit_M, M_xs, M_xm, M_xl), (logit_K, K_xs, K_xm, K_xl), (spatial_M, spatial_K, temporal_M, temporal_K)