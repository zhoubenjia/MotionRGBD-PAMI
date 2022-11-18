'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from .base import Datasets
from torchvision import transforms, set_image_backend
import random, os
from PIL import Image
import numpy as np

def SubSetSampling_func(inputs, reduce=2):
    print('Total training examples', len(inputs))
    sample_dict = {}
    for p, n, l in inputs:
        if l not in sample_dict:
            sample_dict[l] = [(p, n)]
        else:
            sample_dict[l].append((p, n))
    sample_dict = dict([(k, v[::reduce]) for k, v in sample_dict.items()])
    inputs = [(p, n, l) for l, v in sample_dict.items() for p, n in v]
    print('Total training examples after sampling', len(inputs))
    return inputs

class UCFData(Datasets):
    def __init__(self, args, ground_truth, modality, phase='train'):
        super(UCFData, self).__init__(args, ground_truth, modality, phase)

        # sub-set sampling
        # self.inputs = SubSetSampling_func(self.inputs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])

        self.data_path = os.path.join(self.dataset_root, self.inputs[index][0])

        self.clip, skgmaparr = self.image_propose(self.data_path, sl)

        if self.args.Network == 'FusionNet' or self.args.model_ema:
            assert self.typ == 'rgb'
            self.data_path = os.path.join(self.dataset_root, 'nturgb+d_depth_masked', self.inputs[index][0][:-4])
            self.clip1, skgmaparr1 = self.image_propose(self.data_path, sl)
            return (self.clip.permute(0, 3, 1, 2), self.clip1.permute(0, 3, 1, 2)), (skgmaparr, skgmaparr1), \
                   self.inputs[index][2], self.data_path

        return self.clip.permute(0, 3, 1, 2), skgmaparr.permute(0, 3, 1, 2), self.inputs[index][2], self.inputs[index][0]

    def get_path(self, imgs_path, a):
        return os.path.join(imgs_path, "%06d.jpg" % int(a))

    def __len__(self):
        return len(self.inputs)
