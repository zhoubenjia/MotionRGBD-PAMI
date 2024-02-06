'''
This file is modified from:
https://github.com/zhoubenjia/RAAR3DNet/blob/master/Network_Train/lib/datasets/base.py
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend
import torch.nn.functional as F

from PIL import Image
from PIL import ImageFilter, ImageOps
import os, glob
import math, random
import numpy as np
import logging
from tqdm import tqdm as tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import cv2
import json
from scipy.ndimage.filters import gaussian_filter

from timm.data.random_erasing import RandomErasing
# from vidaug import augmentors as va
from .augmentation import *

# import functools
import matplotlib.pyplot as plt  # For graphics
from torchvision.utils import save_image, make_grid
np.random.seed(123)

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class Datasets(Dataset):
    global kpt_dict
    def __init__(self, args, ground_truth, modality, phase='train'):

        self.dataset_root = args.data
        self.sample_duration = args.sample_duration
        self.sample_size = args.sample_size
        self.phase = phase
        self.typ = modality
        self.args = args
        self._w = args.w

        if phase == 'train':
            self.transform = transforms.Compose([ 
            Normaliztion(),
            transforms.ToTensor(), 
            RandomErasing(args.reprob, mode=args.remode, max_count=args.recount, num_splits=0, device='cpu')
            ])
        else:
            self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])

        self.inputs, self.video_apth = self.prepropose(ground_truth)
        
    def prepropose(self, ground_truth, min_frames=16):
        def get_data_list_and_label(data_df):
            return [(lambda arr: (arr[0], int(arr[1]), int(arr[2])))(i.strip().split(' '))
                    for i in open(data_df).readlines()]

        self.inputs = list(filter(lambda x: x[1] > min_frames, get_data_list_and_label(ground_truth)))
        self.inputs = list(self.inputs)
        self.batch_check()
        self.video_apth = dict([(self.inputs[i][0], i) for i in range(len(self.inputs))])
        return self.inputs, self.video_apth
    
    def batch_check(self):
        if self.phase == 'train':
            while len(self.inputs) % (self.args.batch_size * self.args.nprocs) != 0:
                sample = random.choice(self.inputs)
                self.inputs.append(sample)
        else:
            while len(self.inputs) % (self.args.test_batch_size * self.args.nprocs) != 0:
                sample = random.choice(self.inputs)
                self.inputs.append(sample)

    def __str__(self):
        if self.phase == 'train':
            frames = [n[1] for n in self.inputs]
            return 'Training Data Size is: {} \n'.format(len(self.inputs)) + 'Average Train Data frames are: {}, max frames: {}, min frames: {}\n'.format(sum(frames)//len(self.inputs), max(frames), min(frames))
        else:
            frames = [n[1] for n in self.inputs]
            return 'Validation Data Size is: {} \n'.format(len(self.inputs)) + 'Average validation Data frames are: {}, max frames: {}, min frames: {}\n'.format(
                sum(frames) // len(self.inputs), max(frames), min(frames))

    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
            is_flip = True if np.random.uniform(0, 1) < flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def get_path(self, imgs_path, a):
        return os.path.join(imgs_path, "%06d.jpg" % a)

    def depthProposess(self, img):
        h2, w2 = img.shape

        mask = img.copy()
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((10, 10), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find Max Maxtri
        Idx = []
        for i in range(len(contours)):
            Area = cv2.contourArea(contours[i])
            if Area > 500:
                Idx.append(i)
        centers = []

        for i in Idx:
            rect = cv2.minAreaRect(contours[i])
            center, (h, w), degree = rect
            centers.append(center)

        finall_center = np.int0(np.array(centers))
        c_x = min(finall_center[:, 0])
        c_y = min(finall_center[:, 1])
        center = (c_x, c_y)

        crop_x, crop_y = 320, 240
        left = center[0] - crop_x // 2 if center[0] - crop_x // 2 > 0 else 0
        top = center[1] - crop_y // 2 if center[1] - crop_y // 2 > 0 else 0
        crop_w = left + crop_x if left + crop_x < w2 else w2
        crop_h = top + crop_y if top + crop_y < h2 else h2
        rect = (left, top, crop_w, crop_h)
        image = Image.fromarray(img)
        image = image.crop(rect)
        return image

    def image_propose(self, data_path, sl):
        sample_size = self.sample_size
        resize = eval(self.args.resize)
        crop_rect, is_flip = self.transform_params(resize=resize, crop_size=self.args.crop_size, flip=self.args.flip)
        if np.random.uniform(0, 1) < self.args.rotated and self.phase == 'train':
            r, l = eval(self.args.angle)
            rotated = np.random.randint(r, l)
        else:
            rotated = 0

        sometimes = lambda aug: Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq_aug = Sequential([
            RandomResize(self.args.resize_rate),   
            RandomCrop(resize),
            # RandomTranslate(self.args.translate, self.args.translate),
            # sometimes(Salt()),
            # sometimes(GaussianBlur()),
        ])

        def transform(img):
            img = np.asarray(img)
            if img.shape[-1] != 3:
                img = np.uint8(255 * img)
                img = self.depthProposess(img)
                img = cv2.applyColorMap(np.asarray(img), cv2.COLORMAP_JET)
            img = self.rotate(np.asarray(img), rotated)
            img = Image.fromarray(img)
            if self.phase == 'train' and self.args.strong_aug:
                img = self.seq_aug(img)

            img = img.resize(resize)
            img = img.crop(crop_rect)

            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((sample_size, sample_size)))

        def Sample_Image(imgs_path, sl):
            frams = []
            for a in sl:
                ori_image = Image.open(self.get_path(imgs_path, a))
                img = transform(ori_image)
                frams.append(self.transform(img).view(3, sample_size, sample_size, 1))
            if self.args.frp:
                skgmaparr = DynamicImage(frams, dynamic_only=False) #[t, c, h, w]
            else: 
                skgmaparr = torch.ones(*img.shape, 1)
            return torch.cat(frams, dim=3).type(torch.FloatTensor), skgmaparr

        def DynamicImage(frames, dynamic_only): # frames: [[3, 224, 224, 1], ]
            def tensor_arr_rp(arr):
                l = len(arr)
                statics = []
                def tensor_rankpooling(video_arr, lamb=1.):
                    def get_w(N):
                        return [float(i) * 2 - N - 1 for i in range(1, N + 1)]

                    re = torch.zeros(*video_arr[0].size()[:-1])
                    for a, b in zip(video_arr, get_w(len(video_arr))):
                        re += a.squeeze() * b
                    re = (re - re.min()) / (re.max() - re.min())
                    re = np.uint8(255 * np.float32(re.numpy())).transpose(1,2,0)
                    re = self.transform(np.array(re))
                    return re.unsqueeze(-1)

                return [tensor_rankpooling(arr[i:i + self._w]) for i in range(l)]
            arrrp = tensor_arr_rp(frames)
            arrrp = torch.cat(arrrp[:-1], dim=-1).type(torch.FloatTensor)
            return arrrp

        return Sample_Image(data_path, sl)
      

    def get_sl(self, clip):
        sn = self.sample_duration if not self.args.frp else self.sample_duration+1
        if self.phase == 'train':
            f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                   range(int(n * i / sn),
                                                                                         max(int(n * i / sn) + 1,
                                                                                             int(n * (
                                                                                                     i + 1) / sn))))
                           for i in range(sn)]
        else:
            f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                    max(int(
                                                                                                        n * i / sn) + 1,
                                                                                                        int(n * (
                                                                                                                i + 1) / sn))))
                           for i in range(sn)]
        sample_clips = f(int(clip)-self.args.sample_window)
        start = random.sample(range(0, self.args.sample_window), 1)[0]
        if self.phase == 'train':
            return [l + start for l in sample_clips]
        else:
            return f(int(clip))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])

        if self.args.Network == 'FusionNet':
            assert self.typ != 'rgbd', "Please specify '--type rgbd'."
            self.data_path = os.path.join(self.dataset_root, 'rgb', self.inputs[index][0])
            self.clip, skgmaparr = self.image_propose(self.data_path, sl)

            self.data_path = os.path.join(self.dataset_root, 'depth', self.inputs[index][0])
            self.clip1, skgmaparr1 = self.image_propose(self.data_path, sl)
            return (self.clip.permute(0, 3, 1, 2), self.clip1.permute(0, 3, 1, 2)), (skgmaparr, skgmaparr1), \
                   self.inputs[index][2], self.data_path
        
        self.data_path = os.path.join(self.dataset_root, self.typ, self.inputs[index][0])
        self.clip, skgmaparr = self.image_propose(self.data_path, sl)
        
        return self.clip.permute(0, 3, 1, 2), skgmaparr.permute(0, 3, 1, 2), self.inputs[index][2], self.inputs[index][0]
        
    def __len__(self):
        return len(self.inputs)

if __name__ == '__main__':
    import argparse
    from config import Config
    from lib import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='Place config Congfile!')
    parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--nprocs', type=int, default=1)

    parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
    parser.add_argument('--save_output', action='store_true', help='Save logits?')
    parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')

    parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
    parser.add_argument('--save', type=str, default='Checkpoints/', help='experiment name')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    args = parser.parse_args()
    args = Config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dist = False
    args.eval_only = True
    args.test_batch_size = 1

    valid_queue, valid_sampler = build_dataset(args, phase='val')
    for step, (inputs, heatmap, target, _) in enumerate(valid_queue):
        print(inputs.shape)
        input()