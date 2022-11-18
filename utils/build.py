'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
import math
import torch.nn.functional as F
# from .utils import cosine_scheduler
import matplotlib.pyplot as plt
import numpy as np


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

def build_optim(args, model):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate
        )
    return optimizer
#
def build_scheduler(args, optimizer):
    if args.scheduler['name'] == 'cosin':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs-args.scheduler['warm_up_epochs']), eta_min=args.learning_rate_min)
    elif args.scheduler['name'] == 'ReduceLR':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=args.scheduler['patience'], verbose=True,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=3, min_lr=0.00001,
                                                               eps=1e-08)
    else:
        raise NameError('build scheduler error!')

    if args.scheduler['warm_up_epochs'] > 0:
        warmup_schedule = lambda epoch: np.linspace(1e-8, args.learning_rate, args.scheduler['warm_up_epochs'])[epoch]
        return (scheduler, warmup_schedule)
    return (scheduler,)

def build_loss(args):
    loss_Function=dict(
    CE_smooth = LabelSmoothingCrossEntropy(),
    CE = torch.nn.CrossEntropyLoss(),
    MSE = torch.nn.MSELoss(),
    BCE = torch.nn.BCELoss(),
    SoftCE = SoftTargetCrossEntropy(),
    TempLoss = TempoLoss(),
    )
    if args.loss['name'] == 'CE' and args.loss['labelsmooth']:
        return loss_Function['CE_smooth']
    return loss_Function[args.loss['name']]

class SoftTargetCrossEntropy(torch.nn.Module):

    def __init__(self, args=None):
        self.args = args
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # for ii, t in enumerate(target):
        #     v, l = torch.topk(t, k=2, dim=-1)
        #     for i in l:
        #         if i in [0,1,3,8,15,16,17,18]:
        #             target[ii, i] *= 1.5 

        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class TempoLoss(torch.nn.Module):
    def __init__(self):
        super(TempoLoss, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        x: troch.size([b, t, l])
        '''
        loss = 0.0
        for i in range(x.size(1)):
            inp, tar = x[:, i, :], target[:, i, :]
            loss += torch.sum(-tar * F.log_softmax(inp, dim=-1), dim=-1).mean()

        return loss / x.size(1)

class RCM_loss(torch.nn.Module):
    def __init__(self, args, model: torch.nn.Module):
        super(RCM_loss, self).__init__()
        self.args = args
    
    def forward(self, x):
        temp_out, target_out = x
        distill_loss = torch.tensor(0.0).cuda()
        for i, temp_w in enumerate(temp_out):
            # target_weight = self.dtn.multi_scale_transformers[i+3].transformer_enc_media.layers[-1][0].fn.scores
            # target_weight = target_weight.mean(1).mean(1)[:, 1:]
            # target_weight = target_weight.mean(1)[:, 0, 1:]
            # target_weight = self.dtn.multi_scale_transformers[i].class_embedding

            # target_weight = self.dtn.multi_scale_transformers[1][2].layers[i][0].fn.scores
            # # # target_weight = target_weight.mean(1).mean(1)
            # target_weight = target_weight.mean(1).mean(1)[:, 1:]
            target_weight = torch.zeros_like(target_out[0][0])
            for j in range(len(target_out)):
                target_weight += target_out[j][-(len(temp_out)-i)]

            T = self.args.temper
            # distill_loss += F.kl_div(F.log_softmax(temp_w / T, dim=-1),
            #                             F.log_softmax(target_weight.detach() / T, dim=-1),
            #                             reduction='sum')
            # # distill_loss += self.MSE(temp_w, F.softmax(target_weight.detach(), dim=-1))
            target_weight = torch.softmax(target_weight / T, dim=-1)
            distill_loss += torch.sum(-target_weight * F.log_softmax(temp_w / T, dim=-1), dim=-1).mean()
        return distill_loss/len(temp_out)
