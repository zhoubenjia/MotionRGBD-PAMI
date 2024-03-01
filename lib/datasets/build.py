'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from .distributed_sampler import DistributedSampler
from .IsoGD import IsoGDData
from .NvGesture import NvData
from .THU_READ import THUREAD
from .Jester import JesterData
from .NTU import NTUData
from .UCF101 import UCFData
from .base import Datasets
import logging

from torch.utils.data.sampler import  WeightedRandomSampler

def build_dataset(args, phase):
    modality = dict(
        M='rgb',
        K='depth',
        F='Flow',
        rgbd='rgbd'
    )
    assert args.type in modality, 'Error in modality! The currently supported modalities include: M (RGB), K (Depth), F (Flow) and rgbd (RGB-D)'
    Datasets_func = dict(
        basic=Datasets,
        NvGesture=NvData,
        IsoGD=IsoGDData,
        THUREAD=THUREAD,
        Jester=JesterData,
        NTU=NTUData,
    )
    assert args.dataset in Datasets_func, 'Error in dataset Function!'
    if args.local_rank == 0:
        logging.info('Dataset:{}, Modality:{}'.format(args.dataset, modality[args.type]))

    splits = args.splits + '/{}.txt'.format(phase)
    dataset = Datasets_func[args.dataset](args, splits, modality[args.type], phase=phase)
    print(dataset)
    if args.distributed:
        data_sampler = DistributedSampler(dataset)
    else:
        data_sampler = None

    if phase == 'train':
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=(data_sampler is None),
                                                  sampler=data_sampler, pin_memory=True, drop_last=True), data_sampler
    else:
        args.test_batch_size = int(1.5 * args.batch_size)
        return torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                                  shuffle=False,
                                                  sampler=data_sampler, pin_memory=True, drop_last=True), data_sampler
