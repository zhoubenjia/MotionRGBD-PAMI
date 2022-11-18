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
# sys.path.append(os.path.join('..', os.path.abspath(os.path.join(os.getcwd()))) )

from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from timm.optim import create_optimizer
from timm.utils import get_state_dict #, ModelEma, ModelEmaV2

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

from utils.visualizer import Visualizer
from config import Config
from lib import *
from utils import *
#------------------------
# evaluation metrics
#------------------------
from sklearn.decomposition import PCA
from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
from torchvision.utils import save_image, make_grid



def get_args_parser():
    parser = argparse.ArgumentParser('Motion RGB-D training and evaluation script', add_help=False)
    parser.add_argument('--data', type=str, default='/path/to/NTU-RGBD/dataset/', help='data dir')
    parser.add_argument('--splits', type=str, default='/path/to/NTU-RGBD/dataset/dataset_splits/@CS', help='data dir')

    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--test-batch-size', default=4, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)


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
    parser.add_argument('--clip-grad', type=float, default=3., metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
                        
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
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

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

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
    parser.add_argument('--shufflemix', type=float, default=0.0,
                        help='shufflemix alpha, shufflemix enabled if > 0. (default: 0.8)')

    parser.add_argument('--temporal-consist', action='store_true')
    
    return parser

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt.item()

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    seed = args.seed + utils.get_rank()
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    local_rank = utils.get_rank()
    args.nprocs = utils.get_world_size()
    print('nprocs:', args.nprocs)

    
    #----------------------------
    # build function
    #----------------------------
    model = build_model(args)
    model = model.cuda(local_rank)

    train_queue, train_sampler = build_dataset(args, phase='train')
    valid_queue, valid_sampler = build_dataset(args, phase='valid')

    if args.SYNC_BN and args.nprocs > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    criterion = build_loss(args)
    loss_scaler = NativeScaler()
    # optimizer = build_optim(args, model)
    # scheduler = build_scheduler(args, optimizer)
    scheduler, _ = create_scheduler(args, optimizer)
    # scheduler[0].last_epoch = strat_epoch

    if args.finetune:
        load_pretrained_checkpoint(model_without_ddp, args.finetune)

    if args.resume:
        strat_epoch, best_acc = load_checkpoint(model_without_ddp, args.resume, optimizer, scheduler)
        print("Start Epoch: {}, Learning rate: {}, Best accuracy: {}".format(strat_epoch, [g['lr'] for g in
                                                                                                  optimizer.param_groups],
                                                                                    round(best_acc, 4)))
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

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes, args=args)

    model_ema = ModelEma(
            model_without_ddp,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume
            )
    
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args,
        args.out_dim,
        4,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()


    for epoch in range(strat_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        args.distill_lamdb = args.distill
        args.epoch = epoch
        meter_dict_train = train(train_queue, model, criterion, optimizer, epoch, 
                                                        local_rank, loss_scaler, mixup_fn, model_ema, dino_loss)
        scheduler.step(epoch)

        if local_rank == 0:
            logging.info(f'train_loss {round(meter_dict_train["Total_loss"].avg, 4)}')
            state = {'model': model.module.state_dict(),'optimizer': optimizer.state_dict(), 
            'epoch': epoch + 1, 'bestacc': False,
            'scheduler': scheduler.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'model_ema': get_state_dict(model_ema),
            }
            save_checkpoint(state, save=args.save)

            if args.visdom['enable']:
                vis.plot_many({'train_acc': train_acc, 'loss': train_obj,
                               'cosin_similar': meter_dict_train['cosin_similar'].avg}, 'Train-' + args.type, epoch)
                vis.plot_many({'valid_acc': valid_acc, 'loss': valid_obj,
                               'cosin_similar': meter_dict_val['cosin_similar'].avg}, 'Valid-' + args.type, epoch)

def Visfeature(inputs, feature, v_path=None):
    if args.visdom['enable']:
        vis.featuremap('CNNVision',
                       torch.sum(make_grid(feature[0].detach(), nrow=int(feature[0].size(0) ** 0.5), padding=2), dim=0).flipud())
        vis.featuremap('Attention Maps Similarity',
                       make_grid(feature[1], nrow=int(feature[1].detach().cpu().size(0) ** 0.5), padding=2)[0].flipud())

        vis.featuremap('Enhancement Weights', feature[3].flipud())
    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(
            torch.sum(make_grid(feature[0].detach(), nrow=int(feature[0].size(0) ** 0.5), padding=2), dim=0).cpu().numpy(),
            annot=False, fmt='g', ax=ax)
        ax.set_title('CNNVision', fontsize=10)
        fig.savefig(os.path.join(args.save, 'CNNVision.jpg'), dpi=fig.dpi)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(make_grid(feature[1].detach(), nrow=int(feature[1].size(0) ** 0.5), padding=2)[0].cpu().numpy(), annot=False,
                    fmt='g', ax=ax)
        ax.set_title('Attention Maps Similarity', fontsize=10)
        fig.savefig(os.path.join(args.save, 'AttMapSimilarity.jpg'), dpi=fig.dpi)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        sns.heatmap(feature[3].detach().cpu().numpy(), annot=False, fmt='g', ax=ax)
        ax.set_title('Enhancement Weights', fontsize=10)
        fig.savefig(os.path.join(args.save, 'EnhancementWeights.jpg'), dpi=fig.dpi)
        plt.close()

    #------------------------------------------
    # Spatial feature visualization
    #------------------------------------------
    headmap = feature[-1][0][0,:].detach().cpu().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)  # torch.Size([64, 7, 7])
    headmap = torch.from_numpy(headmap)
    img = feature[-1][1]

    result = []
    for map, mg in zip(headmap.unsqueeze(1), img.permute(1,2,3,0)):
        map = cv2.resize(map.squeeze().cpu().numpy(), (mg.shape[0]//2, mg.shape[1]//2))
        map = np.uint8(255 * map)
        map = cv2.applyColorMap(map, cv2.COLORMAP_JET)

        mg = np.uint8(mg.cpu().numpy() * 128 + 127.5)
        mg = cv2.resize(mg, (mg.shape[0]//2, mg.shape[1]//2))
        superimposed_img = cv2.addWeighted(mg, 0.4, map, 0.6, 0)

        result.append(torch.from_numpy(superimposed_img).unsqueeze(0))
    superimposed_imgs = torch.cat(result).permute(0, 3, 1, 2)
    # save_image(superimposed_imgs, os.path.join(args.save, 'CAM-Features.png'), nrow=int(superimposed_imgs.size(0) ** 0.5), padding=2).permute(1,2,0)
    superimposed_imgs = make_grid(superimposed_imgs, nrow=int(superimposed_imgs.size(0) ** 0.5), padding=2).permute(1,2,0)
    cv2.imwrite(os.path.join(args.save, 'CAM-Features.png'), superimposed_imgs.numpy())

    if args.eval_only:
        MHAS_s, MHAS_m, MHAS_l = feature[2]
        MHAS_s, MHAS_m, MHAS_l = MHAS_s.detach().cpu(), MHAS_m.detach().cpu(), MHAS_l.detach().cpu()
        # Normalize
        att_max, index_max = torch.max(MHAS_s.view(MHAS_s.size(0), -1), dim=-1)
        att_min, index_min = torch.min(MHAS_s.view(MHAS_s.size(0), -1), dim=-1)
        MHAS_s = (MHAS_s - att_min.view(-1, 1, 1))/(att_max.view(-1, 1, 1) - att_min.view(-1, 1, 1))

        att_max, index_max = torch.max(MHAS_m.view(MHAS_m.size(0), -1), dim=-1)
        att_min, index_min = torch.min(MHAS_m.view(MHAS_m.size(0), -1), dim=-1)
        MHAS_m = (MHAS_m - att_min.view(-1, 1, 1))/(att_max.view(-1, 1, 1) - att_min.view(-1, 1, 1))

        att_max, index_max = torch.max(MHAS_l.view(MHAS_l.size(0), -1), dim=-1)
        att_min, index_min = torch.min(MHAS_l.view(MHAS_l.size(0), -1), dim=-1)
        MHAS_l = (MHAS_l - att_min.view(-1, 1, 1))/(att_max.view(-1, 1, 1) - att_min.view(-1, 1, 1))

        mhas_s = make_grid(MHAS_s.unsqueeze(1), nrow=int(MHAS_s.size(0) ** 0.5), padding=2)[0]
        mhas_m = make_grid(MHAS_m.unsqueeze(1), nrow=int(MHAS_m.size(0) ** 0.5), padding=2)[0]
        mhas_l = make_grid(MHAS_l.unsqueeze(1), nrow=int(MHAS_l.size(0) ** 0.5), padding=2)[0]
        vis.featuremap('MHAS Map', mhas_l)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(131)
        sns.heatmap(mhas_s.squeeze(), annot=False, fmt='g', ax=ax)
        ax.set_title('\nMHSA Small', fontsize=10)

        ax = fig.add_subplot(132)
        sns.heatmap(mhas_m.squeeze(), annot=False, fmt='g', ax=ax)
        ax.set_title('\nMHSA Medium', fontsize=10)

        ax = fig.add_subplot(133)
        sns.heatmap(mhas_l.squeeze(), annot=False, fmt='g', ax=ax)
        ax.set_title('\nMHSA Large', fontsize=10)
        plt.suptitle('{}'.format(v_path[0].split('/')[-1]), fontsize=20)
        fig.savefig('demo/{}-MHAS.jpg'.format(args.save.split('/')[-1]), dpi=fig.dpi)
        plt.close()

def train(train_queue, model, criterion, optimizer, epoch, local_rank, loss_scaler,
        mixup_fn=None,
        model_ema = None,
        dino_loss=None
        ):
    model.train()

    meter_dict = dict(
        Total_loss=AverageMeter(),
        # CE_loss=AverageMeter(),
        Distil_loss=AverageMeter(),
        # class_loss = AverageMeter(),
        # temporal_loss =  AverageMeter(),
    )
    meter_dict.update(dict(
        cosin_similar=AverageMeter()
    ))
    meter_dict['Data_Time'] = AverageMeter()

    end = time.time()
    CE = SoftTargetCrossEntropy()
    for step, (inputs, heatmap, target, _) in enumerate(train_queue):
        color_samples, depth_samples = inputs
        color_heatmap, depth_heatmap = heatmap
        
        meter_dict['Data_Time'].update((time.time() - end)/args.batch_size)
        color_samples, depth_samples, target, color_heatmap, depth_heatmap = map(lambda x: x.cuda(local_rank, non_blocking=True), 
                                                        [color_samples, depth_samples, target, color_heatmap, depth_heatmap])

        ori_target = target
        ori_color_samples = depth_samples
        if mixup_fn is not None:
        #     color_samples, target = mixup_fn(color_samples, ori_target)
        #     depth_samples, target = mixup_fn(depth_samples, ori_target)
            mixup_alpha = args.mixup
            lam_mix = np.random.beta(mixup_alpha, mixup_alpha)
            calls = ['ShuffleMix', 'ShuffleMix_v1', 'ShuffleMix_v3', 'Vmixup']
            choice = random.choice(calls)
            this_module = sys.modules[__name__]
            getattr(this_module, choice)(color_samples, lam_mix)

        # input exchange
        # if step % 2 == 0:
        #     color_samples,  depth_samples = depth_samples, color_samples
        
        # Teacher
        with torch.no_grad():
            ori_color_samples_flip = ori_color_samples.flip(0)
            (ori_logits, ori_xs, ori_xm, ori_xl, _), _, _ = model_ema(ori_color_samples, color_heatmap)
            (ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip, _), _, _ = model_ema(ori_color_samples_flip, color_heatmap)

            # ori_logits, ori_xs, ori_xm, ori_xl = map(lambda x: torch.softmax(x, dim=-1), [ori_logits, ori_xs, ori_xm, ori_xl])
            # ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip = map(lambda x: torch.softmax(x, dim=-1), [ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip])

            # logits_t = lam_mix * ori_logits + (1. - lam_mix) * ori_logits_flip
            # logits_xs_t = lam_mix * ori_xs + (1. - lam_mix) * ori_xs_flip
            # logits_xm_t = lam_mix * ori_xm + (1. - lam_mix) * ori_xm_flip
            # logits_xl_t = lam_mix * ori_xl + (1. - lam_mix) * ori_xl_flip

        # Student
        (color_logits, cxs, cxm, cxl, _), distillation_loss, feature = model(color_samples, color_heatmap)

        # compute loss
        teacher_output = [[ori_logits, ori_xs, ori_xm, ori_xl], [ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip]]
        student_output = [[color_logits, cxs, cxm, cxl], lam_mix]

        Total_loss = dino_loss(student_output, teacher_output, epoch)

        globals()['Distil_loss'] = distillation_loss * args.distill_lamdb
        Total_loss += Distil_loss
        # globals()['class_loss'] = criterion(color_logits+d_logits, target) * 0.0

        # improve time perception
        # globals()['temporal_loss'] = torch.sum(-torch.softmax(cxs.detach(), dim=-1) * F.log_softmax(cxm, dim=-1), dim=-1).mean()
        # globals()['temporal_loss'] += torch.sum(-torch.softmax(cxs.detach(), dim=-1) * F.log_softmax(cxl, dim=-1), dim=-1).mean()
        # globals()['temporal_loss'] += torch.sum(-torch.softmax(cxm.detach(), dim=-1) * F.log_softmax(cxl, dim=-1), dim=-1).mean()
        # Total_loss += temporal_loss

        globals()['Total_loss'] = Total_loss

        optimizer.zero_grad()
        Total_loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad)
        optimizer.step()
        
        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'cosin' in name:
                meter_dict[name].update(float(feature[2]))

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
                Visfeature(inputs, feature)
        end = time.time()

    torch.cuda.synchronize()
    model_ema.update(model)
    return meter_dict

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
def infer(valid_queue, model, criterion, local_rank, epoch):
    model.eval()

    meter_dict = dict(
        Total_loss=AverageMeter(),
        CE_loss=AverageMeter(),
        Distil_loss=AverageMeter()
    )
    meter_dict.update(dict(
        cosin_similar=AverageMeter(),
    ))
    meter_dict.update(dict(
        Acc_1=AverageMeter(),
        Acc_2=AverageMeter(),
        Acc_3=AverageMeter(),
        Acc=AverageMeter()
    ))

    meter_dict['Infer_Time'] = AverageMeter()
    CE = torch.nn.CrossEntropyLoss()
    grounds, preds, v_paths = [], [], []
    output = {}
    for step, (inputs, heatmap, target, v_path) in enumerate(valid_queue):
        n = inputs.size(0)
        end = time.time()
        inputs, target, heatmap = map(lambda x: x.cuda(local_rank, non_blocking=True), [inputs, target, heatmap])

        (xs, xm, xl, logits), distillation_loss, feature = model(inputs, heatmap)
        meter_dict['Infer_Time'].update((time.time() - end) / n)

        if args.MultiLoss:
            lamd1, lamd2, lamd3, lamd4 = map(float, args.loss_lamdb)
            globals()['CE_loss'] = lamd1 * CE(logits, target) + lamd2 * CE(xs, target) + lamd3 * CE(xm,
                                                                                                    target) + lamd4 * CE(
                xl, target)
        else:
            globals()['CE_loss'] = CE(logits, target)
        globals()['Distil_loss'] = distillation_loss * args.distill_lamdb
        globals()['Total_loss'] = CE_loss + Distil_loss

        grounds += target.cpu().tolist()
        preds += torch.argmax(logits, dim=1).cpu().tolist()
        v_paths += v_path
        torch.distributed.barrier()
        globals()['Acc'] = calculate_accuracy(logits, target)
        globals()['Acc_1'] = calculate_accuracy(xs+xm, target)
        globals()['Acc_2'] = calculate_accuracy(xs+xl, target)
        globals()['Acc_3'] = calculate_accuracy(xl+xm, target)

        for name in meter_dict:
            if 'loss' in name:
                meter_dict[name].update(reduce_mean(globals()[name], args.nprocs))
            if 'cosin' in name:
                meter_dict[name].update(float(feature[2]))
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
                Visfeature(inputs, feature, v_path)

        if args.save_output:
            for t, logit in zip(v_path, logits):
                output[t] = logit
    torch.distributed.barrier()
    grounds_gather = concat_all_gather(torch.tensor(grounds).cuda(local_rank))
    preds_gather = concat_all_gather(torch.tensor(preds).cuda(local_rank))
    grounds_gather, preds_gather = list(map(lambda x: x.cpu().numpy(), [grounds_gather, preds_gather]))

    if local_rank == 0:
        v_paths = np.array(v_paths)
        grounds = np.array(grounds)
        preds = np.array(preds)
        wrong_idx = np.where(grounds != preds)
        v_paths = v_paths[wrong_idx[0]]
        grounds = grounds[wrong_idx[0]]
        preds = preds[wrong_idx[0]]
    return max(meter_dict['Acc'].avg, meter_dict['Acc_1'].avg, meter_dict['Acc_2'].avg, meter_dict['Acc_3'].avg), meter_dict['Total_loss'].avg, dict(grounds=grounds_gather, preds=preds_gather, valid_images=(v_paths, grounds, preds)), meter_dict, output

if __name__ == '__main__':
    # import os
    # args.local_rank=os.environ['LOCAL_RANK']
    parser = argparse.ArgumentParser('Motion RGB-D training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = Config(args)

    try:
        if args.resume:
            args.save = os.path.split(args.resume)[0]
        else:
            args.save = f'{args.save}'
        utils.create_exp_dir(args.save, scripts_to_save=[args.config] + glob.glob('./train.py'))
    except:
        pass
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)