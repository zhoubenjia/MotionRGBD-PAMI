'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import time
import glob
import numpy as np
import shutil
import cv2
import os, random, math
import sys
from pathlib import Path

from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from timm.optim import create_optimizer

import torch
import utils
import logging
import argparse
import traceback
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from collections import OrderedDict

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from utils.visualizer import Visualizer
from config import Config
from lib import *
from utils import *
from timm.utils import get_state_dict #, ModelEma, ModelEmaV2

#------------------------
# evaluation metrics
#------------------------
from sklearn.decomposition import PCA
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
from torchvision.utils import save_image, make_grid
from PIL import Image

from einops import rearrange, repeat

def get_args_parser():
    parser = argparse.ArgumentParser('Motion RGB-D training and evaluation script', add_help=False)
    parser.add_argument('--data', type=str, default='/path/to/NTU-RGBD/dataset/', help='data dir')
    parser.add_argument('--splits', type=str, default='/path/to/NTU-RGBD/dataset/dataset_splits/@CS', help='data dir')
    parser.add_argument('--num-classes', default=None)

    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--test-batch-size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
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
                        help='LR scheduler (default: "cosine", "step", "multistep"')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
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
    parser.add_argument('--decay-milestones', type=list, default=[10, 20, 30], metavar='milestones',
                        help='epoch interval to milestones decay LR, default list[]')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)') 
    
    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
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

    parser.add_argument('--translate', type=int, default=20,
                        help='translate angle (default: 0)')
    parser.add_argument('--strong-aug', action='store_true',
                        help='Strong Augmentation (default: False)')
    parser.add_argument('--resize-rate', type=float, default=0.1,
                        help='random resize rate (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

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
    
    # * Recoupling params
    parser.add_argument('--distill', type=float, default=0.3, metavar='distill param',
                        help='distillation loss coefficient (default: 0.1)')
    parser.add_argument('--temper', type=float, default=0.6, metavar='distillation temperature')

    # * Cross modality loss params
    parser.add_argument('--DC-weight', type=float, default=0.2, metavar='cross depth loss weight')

    # * fp16 params
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')

    # * Rank Pooling params
    parser.add_argument('--frp-num', type=int, default=0, metavar='The Number of Epochs.')
    parser.add_argument('--w', type=int, default=4, metavar='The slide window of FRP.')

    parser.add_argument('--FusionNet', default='cs32', choices=['cs16', 'cs32', 'cs64', 'cv16', 'cv32', 'cv64'],
                        help='used for multi-modal fusion.')
    parser.add_argument('--scc-depth', type=int, default=2, metavar='SCC depth')
    parser.add_argument('--tcc-depth', type=int, default=4, metavar='TCC depth')

    return parser

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt.item()

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    if args.Network != 'FusionNet':
        logging.info('Reset the model to the fusion training state.')
        args.Network = 'FusionNet'

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    if args.amp_opt_level == 'O0':
        logging.info('no apex is used')

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
    model = build_model(args)
    model = model.to(device)

    train_queue, train_sampler = build_dataset(args, phase='train')
    valid_queue, valid_sampler = build_dataset(args, phase='valid')

    if args.SYNC_BN and args.nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes, args=args)
        
    else:
        args.loss['name'] = 'CE'
    
    optimizer = create_optimizer(args, model_without_ddp)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
    criterion = build_loss(args)
    loss_scaler = NativeScaler()
    scheduler, _ = create_scheduler(args, optimizer)

    if args.finetune:
        load_pretrained_checkpoint(model_without_ddp, args.finetune)

    if args.resume:
        strat_epoch, best_acc = load_checkpoint(model_without_ddp, args.resume, optimizer, scheduler)
        print("Start Epoch: {}, Learning rate: {}, Best accuracy: {}".format(strat_epoch, [g['lr'] for g in
                                                                                                  optimizer.param_groups],
                                                                                    round(best_acc, 4)))
        scheduler.step(strat_epoch - 1)
        if args.resumelr:
            for g in optimizer.param_groups:
                args.resumelr = g['lr'] if not isinstance(args.resumelr, float) else args.resumelr
                g['lr'] = args.resumelr
            #resume_scheduler = np.linspace(args.resumelr, 1e-5, args.epochs - strat_epoch)
            resume_scheduler = cosine_scheduler(args.resumelr, 1e-5, args.epochs - strat_epoch + 1, niter_per_ep=1).tolist()
            resume_scheduler.pop(0)

        args.epoch = strat_epoch - 1
    else:
        strat_epoch = 0
        best_acc = 0.0
        args.epoch = strat_epoch
    
    if local_rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        logging.info("learnable param size = %fMB", utils.count_learnable_parameters_in_MB(model))
        if hasattr(model_without_ddp, 'flops'):
            flops = model_without_ddp.flops()
            logging.info(f"number of GFLOPs: {flops / 1e9}")
    
    captuer=None
    if args.FusionNet:
        captuer = FeatureCapter(args, num_classes=args.num_classes)
        captuer = captuer.to(device)
        captuer.eval()

    train_results = dict(
        train_score=[],
        train_loss=[],
        valid_score=[],
        valid_loss=[],
        best_score=0.0
    )

    first_test = True
    if first_test:
        args.distill_lamdb = args.distill
        valid_acc, _, valid_dict, meter_dict, output = infer(valid_queue, model, criterion, local_rank, strat_epoch, device, captuer)
        
        from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
        num_cat = []
        categories = np.unique(valid_dict['grounds'])
        cm = confusion_matrix(valid_dict['grounds'], valid_dict['preds'], labels=categories)
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        # labels, title and ticks
        ax.set_title('Confusion Matrix', fontsize=20)
        ax.set_xlabel('Predicted labels', fontsize=16)
        ax.set_ylabel('True labels', fontsize=16)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.savefig(os.path.join(args.save, "confusion_matrix"), dpi=fig.dpi)

        Accuracy = [(cm[i, i] / sum(cm[i, :])) * 100 if sum(cm[i, :]) != 0 else 0.000001 for i in range(cm.shape[0])]
        Precision = [(cm[i, i] / sum(cm[:, i])) * 100 if sum(cm[:, i]) != 0 else 0.000001 for i in range(cm.shape[1])]
        print('| Class ID \t Accuracy(%) \t Precision(%) |')
        for i in range(len(Accuracy)):
            print('| {0} \t {1} \t {2} |'.format(i, round(Accuracy[i], 2), round(Precision[i], 2)))
        print('-' * 80)
        
        if args.eval_only:
            return

    for epoch in range(strat_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        args.epoch = epoch
        train_acc, train_obj, meter_dict_train = train_one_epoch(train_queue, model, criterion, optimizer, epoch, local_rank, loss_scaler, device, mixup_fn, captuer)
        valid_acc, valid_obj, valid_dict, meter_dict_val, output = infer(valid_queue, model, criterion, local_rank, epoch, device, captuer)
        scheduler.step(epoch)

        if local_rank == 0:
            if valid_acc > best_acc:
                best_acc = valid_acc
                isbest = True
            else:
                isbest = False
            # logging.info(f'train_acc {round(train_acc, 4)}, top-5 {round(meter_dict_train["Acc_top5"].avg, 4)}, train_loss {round(train_obj, 4)}')
            logging.info(f'valid_acc {round(valid_acc, 4)}, best_acc {round(best_acc, 4)}')

            state = {'model': model.module.state_dict(),'optimizer': optimizer.state_dict(), 
            'epoch': epoch + 1, 'bestacc': best_acc,
            'scheduler': scheduler.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'args': args,
            }
            save_checkpoint(state, isbest, args.save)

            train_results['train_score'].append(train_acc)
            train_results['train_loss'].append(train_obj)
            train_results['valid_score'].append(valid_acc)
            train_results['valid_loss'].append(valid_obj)
            train_results['best_score'] = best_acc
            train_results.update(valid_dict)
            train_results['categories'] = np.unique(valid_dict['grounds'])

            if args.visdom['enable']:
                vis.plot_many({'train_acc': train_acc, 'loss': train_obj,
                               'cosin_similar': meter_dict_train['cosin_similar'].avg}, 'Train-' + args.type, epoch)
                vis.plot_many({'valid_acc': valid_acc, 'loss': valid_obj,
                               'cosin_similar': meter_dict_val['cosin_similar'].avg}, 'Valid-' + args.type, epoch)

            if isbest:
                EvaluateMetric(PREDICTIONS_PATH=args.save, train_results=train_results, idx=epoch)
                for k, v in train_results.items():
                    if isinstance(v, list):
                        v.clear()

def train_one_epoch(train_queue, model, criterion, optimizer, epoch, local_rank, loss_scaler, device,
        mixup_fn=None, captuer=None
        ):
    model.train()

    meter_dict = dict(
        Total_loss=AverageMeter(),
    )
    meter_dict['Data_Time'] = AverageMeter()
    meter_dict.update(dict(
        Acc_s=AverageMeter(),
        Acc_m=AverageMeter(),
        Acc_l=AverageMeter(),
        Acc_rgbd=AverageMeter(),
        Acc_rgbd_top5=AverageMeter(),
        Acc_all = AverageMeter(),
        Acc_all_top5=AverageMeter(),
    ))

    meter_dict['Fusion_loss'] = AverageMeter()
    meter_dict['Acc_fusion'] = AverageMeter()
    meter_dict['Acc_fusion_top5'] = AverageMeter()

    end = time.time()
    CE = torch.nn.CrossEntropyLoss()
    MSE = torch.nn.MSELoss()
    rcm_loss = RCM_loss(args, model.module)

    for step, (inputs, heatmap, target, _) in enumerate(train_queue):
        inputs, inputs_aux = inputs
        heatmap, heatmap_aux = heatmap
        inputs_aux, heatmap_aux = map(lambda x: x.to(device, non_blocking=True), [inputs_aux, heatmap_aux])
        
        meter_dict['Data_Time'].update((time.time() - end)/args.batch_size)
        inputs, target, heatmap = map(lambda x: x.to(device, non_blocking=True), [inputs, target, heatmap])

        ori_target, target_aux = target, target
        if mixup_fn is not None:
            inputs, target = mixup_fn(inputs, target)
            inputs_aux, target_aux = mixup_fn(inputs_aux, ori_target)
        
        images = inputs
        with torch.no_grad():
            (logits, xs, xm, xl), (logit_K, K_xs, K_xm, K_xl), hidden_feature = captuer(inputs, inputs_aux)
        output, _ = model(hidden_feature)
        fusion_loss = criterion(output[0], target) + criterion(output[1], target_aux)

        globals()['Fusion_loss'] = fusion_loss
        Total_loss = fusion_loss
        globals()['Acc_fusion'], globals()['Acc_fusion_top5'] = accuracy((output[0] + output[1])/2.0, ori_target, topk=(1, 5))

        if args.ACCUMULATION_STEPS > 1:
            globals()['Total_loss']  = Total_loss / args.ACCUMULATION_STEPS

            Total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if (step + 1) % args.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            globals()['Total_loss']  = Total_loss
            
            optimizer.zero_grad()
            if args.fp16:
                with amp.scale_loss(Total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_grad)
            else:
                Total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        
        #---------------------
        # Meter performance
        #---------------------
        torch.distributed.barrier()
        globals()['Acc_rgbd'], globals()['Acc_rgbd_top5'] = accuracy(logits+logit_K, ori_target, topk=(1, 5))
        globals()['Acc_all'], globals()['Acc_all_top5'] = accuracy((output[0] + output[1] + logits + logit_K)/4.0, ori_target, topk=(1, 5))
        globals()['Acc_s'], _ = accuracy(xs+K_xs, ori_target, topk=(1, 5))
        globals()['Acc_m'], _ = accuracy(xm+K_xm, ori_target, topk=(1, 5))
        globals()['Acc_l'], _ = accuracy(xl+K_xl, ori_target, topk=(1, 5))

        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'Acc' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': '{}/{}'.format(epoch + 1, args.epochs),
                'Mini-Batch': '{:0>5d}/{:0>5d}'.format(step + 1,
                                                       len(train_queue.dataset) // (args.batch_size * args.nprocs)),
                'Lr': optimizer.param_groups[0]["lr"],
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)

            if args.vis_feature:
                Visfeature(args, model.module, images, weight_softmax=torch.softmax(logits, dim=-1), FusionNet=True)
        end = time.time()

    if local_rank == 0:
        print('*'*20)
        print_func(dict([(name,  meter_dict[name].avg) for name in meter_dict]))
        print('*'*20)

    return meter_dict['Acc_all'].avg, meter_dict['Total_loss'].avg, meter_dict

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def infer(valid_queue, model, criterion, local_rank, epoch, device, captuer, obtain_softmax_score=True):
    model.eval()
        
    meter_dict = dict(
        Total_loss=AverageMeter(),
    )
    meter_dict.update(dict(
    Acc_r=AverageMeter(),
    Acc_r_top5=AverageMeter(),
    Acc_d=AverageMeter(),
    Acc_d_top5=AverageMeter(),
    Acc_rgbd = AverageMeter(),
    Acc_rgbd_top5=AverageMeter(),
    Acc_fusion=AverageMeter(),
    Acc_fusion_top5=AverageMeter(),
    ))

    meter_dict['Infer_Time'] = AverageMeter()
    CE = torch.nn.CrossEntropyLoss()
    MSE = torch.nn.MSELoss()
    grounds, preds, v_paths = [], {0:[], 1:[]}, []
    logits_out = {}
    softmax_score = {}
    embedding_dict = OrderedDict()
    for step, (inputs, heatmap, target, v_path) in enumerate(valid_queue):
        inputs, inputs_aux = inputs
        heatmap, heatmap_aux = heatmap
        inputs_aux, heatmap_aux = map(lambda x: x.to(device, non_blocking=True), [inputs_aux, heatmap_aux])
            
        n = inputs.size(0)
        end = time.time()
        inputs, target, heatmap = map(lambda x: x.to(device, non_blocking=True), [inputs, target, heatmap])

        images = inputs
        (logits, xs, xm, xl), (logit_K, K_xs, K_xm, K_xl), hidden_feature = captuer(inputs, inputs_aux)
        output, temp_out = model(hidden_feature)

        Fusion_loss = CE(output[0], target) + CE(output[1], target)
        Total_loss = Fusion_loss

        globals()['Total_loss'] = Total_loss 
        meter_dict['Infer_Time'].update((time.time() - end) / n)

        v_paths += v_path
        grounds += target.cpu().tolist()

        # save logits from outputs
        preds[0] += torch.argmax((logits+logit_K)/2., dim=1).cpu().tolist()
        preds[1] += torch.argmax((logits + logit_K + output[0] + output[1])/4., dim=1).cpu().tolist()

        torch.distributed.barrier()
        globals()['Acc_r'], globals()['Acc_r_top5'] = accuracy(logits, target, topk=(1, 5))
        globals()['Acc_d'], globals()['Acc_d_top5'] = accuracy(logit_K, target, topk=(1, 5))
        globals()['Acc_rgbd'], globals()['Acc_rgbd_top5'] = accuracy((logits+logit_K)/2., target, topk=(1, 5))
        globals()['Acc_fusion'], globals()['Acc_fusion_top5'] = accuracy((logits + logit_K + output[0] + output[1])/4., target, topk=(1, 5))

        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'Acc' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))

        if step % args.report_freq == 0 and local_rank == 0:
            log_info = {
                'Epoch': epoch + 1,
                'Mini-Batch': '{:0>4d}/{:0>4d}'.format(step + 1, len(valid_queue.dataset) // (
                            args.test_batch_size * args.nprocs)),
            }
            log_info.update(dict((name, '{:.4f}'.format(value.avg)) for name, value in meter_dict.items()))
            print_func(log_info)
            if args.vis_feature:
                Visfeature(args, model.module, images, v_path, torch.softmax(logits, dim=-1), FusionNet=True)

    # select best acc output
    acc_list = torch.tensor([meter_dict['Acc_fusion'].avg, meter_dict['Acc_rgbd'].avg])
    best_idx = torch.argmax(acc_list).tolist()
    preds = preds[best_idx] # Note: only preds be refined


    grounds_gather = concat_all_gather(torch.tensor(grounds).to(device))
    preds_gather = concat_all_gather(torch.tensor(preds).to(device))
    grounds_gather, preds_gather = list(map(lambda x: x.cpu().numpy(), [grounds_gather, preds_gather]))

    if local_rank == 0:
        print('*'*20)
        print_func(dict([(name,  meter_dict[name].avg) for name in meter_dict]))
        print('*'*20)
        v_paths = np.array(v_paths)
        grounds = np.array(grounds)
        preds = np.array(preds)
        wrong_idx = np.where(grounds != preds)
        v_paths = v_paths[wrong_idx[0]]
        grounds = grounds[wrong_idx[0]]
        preds = preds[wrong_idx[0]]

    return  acc_list.tolist()[best_idx], meter_dict['Total_loss'].avg, dict(grounds=grounds_gather, preds=preds_gather, valid_images=(v_paths, grounds, preds)), meter_dict, logits_out

if __name__ == '__main__':
    # import os
    # args.local_rank=os.environ['LOCAL_RANK']
    parser = argparse.ArgumentParser('Motion RGB-D training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = Config(args)

    if args.save and args.local_rank == 0:
        Path(args.save).mkdir(parents=True, exist_ok=True)

    try:
        if args.resume:
            args.save = os.path.split(args.resume)[0]
        else:
            args.save = f'{args.save}'
        utils.create_exp_dir(args.save, scripts_to_save=[args.config] + glob.glob('./train.py') + glob.glob('lib/model/*.py'))
    except:
        pass
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)