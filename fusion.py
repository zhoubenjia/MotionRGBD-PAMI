'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import os, random, math
import time
import glob
import numpy as np
import shutil

import torch

import logging
import argparse
import traceback
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join("..", os.getcwd())))
from config import Config
from lib import *
import torch.distributed as dist
from utils import *
from utils.build import *

from lib.model.DSN_v2 import DSNNetV2

parser = argparse.ArgumentParser('Motion RGB-D training and evaluation script', add_help=False)
parser.add_argument('--data', type=str, default='/path/to/NTU-RGBD/dataset/', help='data dir')
parser.add_argument('--splits', type=str, default='/path/to/NTU-RGBD/dataset/dataset_splits/@CS', help='data dir')

parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--test-batch-size', default=32, type=int)
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')

parser.add_argument('--config', help='Load Congfile.')
parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
parser.add_argument('--local_rank', type=int, default=0)
# parser.add_argument('--nprocs', type=int, default=1)
parser.add_argument('--type', default='M',
                    help='data types, e.g., "M" or "K"')


parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
parser.add_argument('--save_output', action='store_true', help='Save logits?')
parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')
parser.add_argument('--resume', default='', help='resume from checkpoint')

# * Finetuning params
parser.add_argument('--finetune', default='', help='finetune from checkpoint')

parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--save', type=str, default='Checkpoints/', help='experiment dir')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

parser.add_argument('--shuffle', default=False, action='store_true', help='Tokens shuffle')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=5., metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0005,
                    help='weight decay (default: 0.0005)')
parser.add_argument('--ACCUMULATION-STEPS', type=int, default=0,
                    help='accumulation step (default: 0.0)')
                    
# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

parser.add_argument('--mixup-dynamic', action='store_true', default=False, help='')

parser.add_argument('--model-ema', default=True)
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Augmentation parameters
parser.add_argument('--autoaug', action='store_true')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

parser.add_argument('--repeated-aug', action='store_true')
parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
parser.set_defaults(repeated_aug=True)

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# Vision Transformer
parser.add_argument('--model', type=str, default='deit_tiny_patch16_224')

    # * ShuffleMix params
parser.add_argument('--shufflemix', type=float, default=0.2,
                    help='shufflemix alpha, shufflemix enabled if > 0. (default: 0.0)')
parser.add_argument('--smixmode', type=str, default='sm',
                    help='ShuffleMix strategies (default: "shufflemix(sm)", Per "sm_v1", "sm_v2", or "sm_v3", "mu_sm")')
parser.add_argument('--smprob', type=float, default=0.3, metavar='ShuffleMix Prob',
                    help='ShuffleMix enable prob (default: 0.3)')

parser.add_argument('--temporal-consist', action='store_true')
parser.add_argument('--tempMix', action='store_true')
parser.add_argument('--MixIntra', action='store_true')
parser.add_argument('--replace-prob', type=float, default=0.25, metavar='MixIntra replace Prob')

# DTN example sampling params
parser.add_argument('--sample-duration', type=int, default=16,
                    help='The sampled frames in a video.')
parser.add_argument('--intar-fatcer', type=int, default=2,
                    help='The sampled frames in a video.')
parser.add_argument('--sample-window', type=int, default=1,
                        help='Range of frames sampling (default: 1)')
parser.add_argument('--translate', type=int, default=0,
                        help='translate angle (default: 0)')

# * Recoupling params
parser.add_argument('--distill', type=float, default=0.3, metavar='distill param',
                    help='distillation loss coefficient (default: 0.1)')
parser.add_argument('--temper', type=float, default=0.6, metavar='distillation temperature')

# * Cross modality loss params
parser.add_argument('--DC-weight', type=float, default=0.5, metavar='cross depth loss weight')

# * Rank Pooling params
parser.add_argument('--frp-num', type=int, default=0, metavar='The Number of Epochs.')
parser.add_argument('--w', type=int, default=4, metavar='The slide window of FRP.')

# * fp16 params
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')

parser.add_argument('--FusionNet', default=False)
                    
args = parser.parse_args()
args = Config(args)

class FusionModule(nn.Module):
    def __init__(self, args, fusion='add'):
        super(FusionModule, self).__init__()
        self.fusion = fusion
        self.args = args
        build_model(args)
        self.rgb = DSNNetV2(args, num_classes=args.num_classes, pretrained=args.pretrained)
        self.depth = DSNNetV2(args, num_classes=args.num_classes, pretrained=args.pretrained)

        rgb_checkpoint = args.rgb_checkpoint[args.FusionNet]
        self.strat_epoch_r, self.best_acc_r = load_checkpoint(self.rgb, rgb_checkpoint)
        print(f'Best acc RGB: {self.best_acc_r}')
        depth_checkpoint = args.depth_checkpoint[args.FusionNet]
        self.strat_epoch_d, self.best_acc_d = load_checkpoint(self.depth, depth_checkpoint)
        print(f'Best acc depth: {self.best_acc_d}')

    def forward(self, r, d):
        self.args.epoch = self.strat_epoch_r - 1
        (r_x, r_xs, r_xm, r_xl), _ = self.rgb(r)
        self.args.epoch = self.strat_epoch_d - 1
        (d_x, xs, xm, xl), _ = self.depth(d)

        distance = F.pairwise_distance(r_x, d_x, p=2)

        if self.fusion == 'add':
            x = (r_x + d_x) / 2.
        else:
            x = r_x * d_x
        return x, distance
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt.item()

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    seed = args.seed + utils.get_rank()
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    local_rank = utils.get_rank()
    args.nprocs = utils.get_world_size()
    print('nprocs:', args.nprocs)
    device = torch.device(args.device)

    #----------------------------
    # build function
    #----------------------------
    model = FusionModule(args)
    model = model.to(device)

    valid_queue, valid_sampler = build_dataset(args, phase='valid')

    criterion = build_loss(args)

    if args.SYNC_BN and args.nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module

    print("param size = %fMB"%utils.count_parameters_in_MB(model))

    valid_dict = infer(valid_queue, model, criterion, local_rank, device)

@torch.no_grad()
def infer(valid_queue, model, criterion, local_rank, device, epoch=0):
    model.eval()

    meter_dict = dict(
        CE_loss=AverageMeter(),
    )
    meter_dict.update(dict(
        Acc=AverageMeter(),
        Acc_top5=AverageMeter(),
    ))
    meter_dict['distance'] = AverageMeter()

    meter_dict['Infer_Time'] = AverageMeter()
    CE = torch.nn.CrossEntropyLoss()
    MSE = torch.nn.MSELoss()
    grounds, preds, v_paths = [], [], []
    output = {}
    for step, (inputs, heatmap, target, v_path) in enumerate(valid_queue):

        color, depth = inputs
        color, depth, target = map(lambda x: x.to(device, non_blocking=True), [color, depth, target])

        features = []
        def hook(module, input, output): 
            features.append(output.clone().detach())
        handle = model.module.rgb.dtn.multi_scale_transformers[0][2].register_forward_hook(hook)
        handle = model.module.rgb.dtn.multi_scale_transformers[1][2].register_forward_hook(hook)
        handle = model.module.rgb.dtn.multi_scale_transformers[2][2].register_forward_hook(hook)
        handle = model.module.depth.dtn.multi_scale_transformers[0][2].register_forward_hook(hook)
        handle = model.module.depth.dtn.multi_scale_transformers[1][2].register_forward_hook(hook)
        handle = model.module.depth.dtn.multi_scale_transformers[2][2].register_forward_hook(hook)
        # handle1.remove()
            
        n = target.size(0)
        end = time.time()
        output, distance = model(color, depth)
        distance = F.pairwise_distance(features[0][:, 0]+features[1][:, 0]+features[2][:, 0], features[3][:, 0]+features[4][:, 0]+features[5][:, 0], p=2).mean()

        globals()['CE_loss'] = CE(output, target)
        globals()['distance'] = distance.mean()
        meter_dict['Infer_Time'].update((time.time() - end) / n)
        grounds += target.cpu().tolist()
        preds += torch.argmax(output, dim=1).cpu().tolist()
        v_paths += v_path
        torch.distributed.barrier()
        globals()['Acc'], globals()['Acc_top5'] = accuracy(output, target, topk=(1, 5))

        for name in meter_dict:
            if 'Time' not in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': epoch + 1,
                'Mini-Batch': '{:0>4d}/{:0>4d}'.format(step + 1, len(valid_queue.dataset) // (
                            args.test_batch_size * args.nprocs)),
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)
            
    torch.distributed.barrier()
    grounds_gather = concat_all_gather(torch.tensor(grounds).to(device))
    preds_gather = concat_all_gather(torch.tensor(preds).to(device))
    grounds_gather, preds_gather = list(map(lambda x: x.cpu().numpy(), [grounds_gather, preds_gather]))

    print(dict([(name,  meter_dict[name].avg) for name in meter_dict]))
    
    return meter_dict


if __name__ == '__main__':
    try:
        main(args)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()