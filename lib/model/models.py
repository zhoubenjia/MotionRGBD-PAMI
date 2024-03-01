""" 
This file is modified from: 
https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/deit.py
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as nnf

from torchvision.utils import save_image, make_grid
import numpy as np
import cv2

import random
random.seed(123)

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.resnet import Bottleneck, ResNet
from timm.models.resnet import _cfg as _cfg_resnet
from timm.models.helpers import build_model_with_cfg

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

def TokensCutOff(x, tua = 0.4):
        CLS, DIS = x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1)
        tokens = x[:, 2:, :]
        B, N, C = tokens.shape

        mask = torch.ones(B, N, requires_grad=False).cuda()
        prob = torch.rand(B, N, requires_grad=False).cuda()
        mask = torch.where(prob > tua, mask, torch.full_like(mask, 1e-8))
        TokenMask = mask.view(B, N, 1).expand_as(tokens)

        x = tokens * TokenMask
        x = torch.cat((CLS, DIS, x), dim=1)
        return x

def FeatureCutOff(x, tua = 0.4):
        CLS, DIS = x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1)
        tokens = x[:, 2:, :]
        B, N, C = tokens.shape

        mask = torch.ones(B, C, requires_grad=False).cuda()
        prob = torch.rand(B, C, requires_grad=False).cuda()
        mask = torch.where(prob > tua, mask, torch.full_like(mask, 1e-8))
        TokenMask = mask.view(B, 1, C).expand_as(tokens)
        
        x = tokens * TokenMask
        x = torch.cat((CLS, DIS, x), dim=1)
        return x

def shuffle_unit(features, shift, group, begin=0, return_idex=False):
        batchsize = features.size(0)
        dim = features.size(-1)
        labels = torch.arange(0, features.size(-2), 1, device=features.device).expand(batchsize, -1)

        # Shift Operation
        feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
        labels = torch.cat([labels[:, begin-1+shift:], labels[:, begin:begin-1+shift]], dim=1)
        x = feature_random

        # Patch Shuffle Operation
        x = x.view(batchsize, group, -1, dim)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, dim)

        labels = labels.view(batchsize, group, -1, 1)
        labels = torch.transpose(labels, 1, 2).contiguous()
        labels = labels.view(batchsize, -1)

        if return_idex:
            return x, labels
        return x

def random_shuffle_unit(features, return_idex=False, batch_premutation=False, sort_label=None):
    if sort_label:
        B, N, C = features.shape
        labels = []
        perms_idx = []
        for b in range(B):
            perm_idx = random.choice(list(sort_label.keys()))
            label = sort_label[perm_idx]
            perms_idx.append(perm_idx + b * N)
            labels.append(label)
        perms_idx = torch.cat(perms_idx)
        x = features.contiguous().view(-1, C)
        x = x[perms_idx, :]
        x = x.view(B, N, C)

        if return_idex:
            return x, torch.tensor(labels,  device=features.device), perms_idx

    if batch_premutation:
        B, N, C = features.shape
        labels = torch.arange(0, N, 1, device=features.device)
        # labels = (labels - labels.min())/(labels.max() - labels.min()) + 1e-8
        labels = labels.expand(B, -1)

        # perturbation = torch.rand([B, N], device=features.device) - torch.rand([B, N], device=features.device)
        # labels = labels + perturbation

        index = torch.cat([torch.randperm(N) + b * N for b in range(B)], dim=0)
        x = features.contiguous().view(-1, C)
        x = x[index, :]
        x = x.view(B, N, C)
        labels = labels.contiguous().view(-1)[index].view(B, -1)

    else:
        batchsize = features.size(0)
        dim = features.size(-1)
        num_patch = features.size(-2)

        labels = torch.arange(0, features.size(-2), 1, device=features.device)
        # labels = (labels - labels.min())/(labels.max() - labels.min()) + 1e-8
        labels = labels.expand(batchsize, -1)

        # perturbation = torch.rand([B, N]) - torch.rand([B, N])
        # labels = labels + perturbation

        index = torch.randperm(features.size(-2))
        labels = labels[:, index]
        x = features[:, index, :]

    if return_idex:
        return x, labels, index
    return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None

    def forward(self, x):
        xori = x
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn(self):
        return self.attn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = [drop, drop]

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):

    def __init__(self, args, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self._args = kwargs['args']
        del kwargs['args']

        super().__init__(*args, **kwargs)

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(
        #         dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'], qkv_bias=kwargs['qkv_bias'], drop=kwargs['drop_rate'],
        #         attn_drop=kwargs['attn_drop_rate'], drop_path=dpr[i], norm_layer=kwargs['norm_layer'], act_layer=kwargs['act_layer'])
        #     for i in range(kwargs['depth'])])

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

        self.shuffle = self._args.shuffle
        self.Token_cutoff = self._args.Token_cutoff
        self.tua_token = self._args.tua_token

        self.Feature_cutoff = self._args.Feature_cutoff
        self.tua_feature = self._args.tua_feature

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.shuffle and self.training:
            CLS, DIS = x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1)
            x = shuffle_unit(x[:, 2:, :], shift=8, group=2)
            x = torch.cat((CLS, DIS, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
            if self.Token_cutoff and self.training:
                x = TokensCutOff(x, self.tua_token)
            if self.Feature_cutoff and self.training:
                x = FeatureCutOff(x, self.tua_feature)
        
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

class Video2Image(nn.Module):
    def __init__(self, inp_channel=16):
        super(Video2Image, self).__init__()
        # self.MLP = nn.Sequential(
        #     # input: [B, N, C]
        #     nn.Linear(C, C//2),
        #     nn.ReLU(),
        #     nn.Linear(C//2, C)
        # )
        self.channel1 = nn.Conv2d(inp_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.channel2 = nn.Conv2d(inp_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.channel3 = nn.Conv2d(inp_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=False)

        self.channel1_reverse = nn.Conv2d(1, inp_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.channel2_reverse = nn.Conv2d(1, inp_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.channel3_reverse = nn.Conv2d(1, inp_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reverse = nn.BatchNorm3d(3)
        self.relu_reverse = nn.ReLU(inplace=False)

        self.compressed = None
    
    def get_compressed_img(self):
        return self.compressed

    def forward(self, x):
        B, C, T, H, W = x.shape
        # x = rearrange(x, 'b c t h w -> (b c) t h w)')
        x_channel1 =  self.channel1(x[:, 0, :, :, :])
        x_channel2 =  self.channel2(x[:, 1, :, :, :])
        x_channel3 =  self.channel3(x[:, 2, :, :, :])
        x = torch.cat((x_channel1, x_channel2, x_channel3), dim=1)
        x = self.relu(self.bn(x))
        self.compressed = x
        
        x_channel1_reverse =  self.channel1_reverse(x[:, 0, :, :].unsqueeze(1))
        x_channel2_reverse =  self.channel2_reverse(x[:, 1, :, :].unsqueeze(1))
        x_channel3_reverse =  self.channel3_reverse(x[:, 2, :, :].unsqueeze(1))
        x_reverse = torch.cat((x_channel1_reverse.unsqueeze(1), x_channel2_reverse.unsqueeze(1), x_channel3_reverse.unsqueeze(1)), dim=1)
        x_reverse = self.relu_reverse(self.bn_reverse(x_reverse))
        return x, x_reverse

class VisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self._args = kwargs['args']
        del kwargs['args']
        super().__init__(*args, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                self._args, dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'], qkv_bias=kwargs['qkv_bias'], drop=kwargs['drop_rate'],
                drop_path=dpr[i], norm_layer=kwargs['norm_layer'])
            for i in range(kwargs['depth'])])
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.video2Img = Video2Image(self._args.sample_duration)
    
    def get_cls_token(self):
        return self.CLSToken
    def get_patch_token(self):
        return self.PatchToken

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        self.PatchToken = x[:, 1:]

        return self.pre_logits(x[:, 0])
        
    def forward(self, x): 
        # x.size: torch.Size([16, 3, 16, 224, 224])
        x, x_reverse = self.video2Img(x)
        x = self.forward_features(x)
        self.CLSToken = x
        x = self.head(x)
        return x, x_reverse


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def _create_resnet(variant, pretrained=False, **kwargs):
    del kwargs['args']
    return build_model_with_cfg(ResNet, variant, pretrained, default_cfg=_cfg_resnet(), **kwargs)

@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    model = _create_resnet('resnet50', pretrained, **model_args)
    # model.default_cfg = _cfg_resnet()
    return model

@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    model = _create_resnet('resnet101', pretrained, **model_args)
    return model