'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

from .DSN import DSNNet
from .DSN_v2 import DSNNetV2
from .fusion_Net import CrossFusionNet, SFNNet
from .models import *

from timm.models import create_model

import logging

def build_model(args):        
    num_classes = dict(
        IsoGD=249,
        NvGesture=25,
        Jester=27,
        THUREAD=40,
        NTU=60,
        UCF101=101
    )
    if args.num_classes is not None:
        num_classes[args.dataset] = args.num_classes

    func_dict = dict(
        DSN=DSNNet,
        DSNV2=DSNNetV2,
        FusionNet=CrossFusionNet
    )
    assert args.dataset in num_classes, 'Error in load dataset !'
    assert args.Network in func_dict, 'Error in Network function !'
    args.num_classes = num_classes[args.dataset]

    if args.local_rank == 0:
        logging.info('Model:{}, Total Categories:{}'.format(args.Network, args.num_classes))

    return func_dict[args.Network](args, num_classes=args.num_classes, pretrained=args.pretrained)
