'''
This file is modified from:
https://github.com/yuhuixu1993/PC-DARTS/blob/master/utils.py
'''

import os
import numpy as np
import torch
import torch.distributed as dist
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
import random
from .build import SoftTargetCrossEntropy

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
import cv2

from einops import rearrange, repeat

class ClassAcc():
    def __init__(self, GESTURE_CLASSES):
        self.class_acc = dict(zip([i for i in range(GESTURE_CLASSES)], [0]*GESTURE_CLASSES))
        self.single_class_num = [0]*GESTURE_CLASSES
    def update(self, logits, target):
        pred = torch.argmax(logits, dim=1)
        for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
            if p == t:
                self.class_acc[t] += 1
            self.single_class_num[t] += 1
    def result(self):
        return [round(v / (self.single_class_num[k]+0.000000001), 4) for k, v in self.class_acc.items()]
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def adjust_learning_rate(optimizer, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    df = 0.7
    ds = 40000.0
    lr = lr * np.power(df, step / ds)
    # lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# def accuracy(output, target, topk=(1,)):
#   maxk = max(topk)
#   batch_size = target.size(0)

#   _, pred = output.topk(maxk, 1, True, True)
#   pred = pred.t()
#   correct = pred.eq(target.view(1, -1).expand_as(pred))

#   res = []
#   for k in topk:
#     correct_k = correct[:k].view(-1).float().sum(0)
#     res.append(correct_k.mul_(100.0/batch_size))
#   return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        #n_correct_elems = correct.float().sum().data[0]
        # n_correct_elems = correct.float().sum().item()
    # return n_correct_elems / batch_size
    return correct_k.mul_(1.0 / batch_size)

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_learnable_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if v.requires_grad)/1e6

def save_checkpoint(state, is_best=False, save='./', filename='checkpoint.pth.tar'):
  filename = os.path.join(save, filename)
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def load_checkpoint(model, model_path, optimizer=None, scheduler=None):
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(4))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    bestacc = checkpoint['bestacc']
    return epoch, bestacc

def load_pretrained_checkpoint(model, model_path):
    # params = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(local_rank))['model']
    params = torch.load(model_path, map_location='cpu')['model']
    new_state_dict = OrderedDict()

    for k, v in params.items():
        name = k[7:] if k[:7] == 'module.' else k
        # if name not in ['dtn.mlp_head_small.2.bias', "dtn.mlp_head_small.2.weight",
        #                 'dtn.mlp_head_media.2.bias', "dtn.mlp_head_media.2.weight",
        #                 'dtn.mlp_head_large.2.bias', "dtn.mlp_head_large.2.weight"]:

        # if v.shape == model.state_dict()[name].shape:
        try:
            if v.shape == model.state_dict()[name].shape and name not in ['dtn.multi_scale_transformers.0.3.2.weight', 'dtn.multi_scale_transformers.0.3.2.bias', 'dtn.multi_scale_transformers.1.3.2.weight', 'dtn.multi_scale_transformers.1.3.2.bias', 'dtn.multi_scale_transformers.2.3.2.weight', 'dtn.multi_scale_transformers.2.3.2.bias']:
                new_state_dict[name] = v
        except:
            continue
    ret = model.load_state_dict(new_state_dict, strict=False)
    print('Missing keys: \n', ret.missing_keys)
    # return model

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    if not os.path.exists(os.path.join(path, 'scripts')):
      os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
  
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def uniform_sampling(clips_num, sn, random=True):
    if random:
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
    return f(clips_num)

class DINOLoss(torch.nn.Module):
    def __init__(self, args, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.args = args
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        self.CE = SoftTargetCrossEntropy()

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        [ori_logits, ori_xs, ori_xm, ori_xl], [ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip] = teacher_output
        [color_logits, cxs, cxm, cxl], lam_mix = student_output

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        ori_logits, ori_xs, ori_xm, ori_xl = map(lambda x: torch.softmax((x - self.center) / temp, dim=-1), [ori_logits, ori_xs, ori_xm, ori_xl])
        ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip = map(lambda x: torch.softmax((x - self.center) / temp , dim=-1), [ori_logits_flip, ori_xs_flip, ori_xm_flip, ori_xl_flip])

        logits_t = lam_mix * ori_logits + (1. - lam_mix) * ori_logits_flip
        logits_xs_t = lam_mix * ori_xs + (1. - lam_mix) * ori_xs_flip
        logits_xm_t = lam_mix * ori_xm + (1. - lam_mix) * ori_xm_flip
        logits_xl_t = lam_mix * ori_xl + (1. - lam_mix) * ori_xl_flip

        # color_logits, cxs, cxm, cxl = map(lambda x: torch.softmax(x / self.student_temp, dim=-1), [color_logits, cxs, cxm, cxl])
        color_logits, cxs, cxm, cxl = map(lambda x: x / self.student_temp, [color_logits, cxs, cxm, cxl])

        Total_loss = 0.0
        CE = self.CE
        if self.args.MultiLoss:
            lamd1, lamd2, lamd3, lamd4 = map(float, self.args.loss_lamdb)
            CE_loss = lamd1*CE(color_logits, logits_t) + lamd2*CE(cxs, logits_xs_t) + \
                                   lamd3*CE(cxm, logits_xm_t) + lamd4*CE(cxl, logits_xl_t)
        else:
            CE_loss = CE(color_logits, logits_t)
        Total_loss += CE_loss

        self.update_center(logits_t)
        return Total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


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

def Visfeature(args, model, inputs, v_path=None, weight_softmax=None, FusionNet=False):
    # TSNE cluster
    if FusionNet:
        # pca_data = model.pca_data.detach().cpu()
        # targets = model.target_data.cpu()
        pca_data, targets = model.get_cluster_visualization()

        tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        data = pd.DataFrame(pca_data.cpu().numpy())
        data_pca = tsne.fit_transform(data)

        data.insert(0, 'label', pd.DataFrame(targets.cpu().numpy()))

        fig, ax = plt.subplots()
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=data['label'], s=25, cmap='rainbow', 
        alpha=0.8, edgecolors='none')

        legend1 = ax.legend(*scatter.legend_elements(fmt="{x:.0f}"),
                    loc="best", title="Feature Type")
        ax.add_artist(legend1)

        fig.savefig(args.save + '/cluster-result.png')
        plt.close()
        return
        
    if args.visdom['enable']:
        vis.featuremap('CNNVision',
                       torch.sum(make_grid(feature[0].detach(), nrow=int(feature[0].size(0) ** 0.5), padding=2), dim=0).flipud())
        vis.featuremap('Attention Maps Similarity',
                       make_grid(feature[1], nrow=int(feature[1].detach().cpu().size(0) ** 0.5), padding=2)[0].flipud())

        vis.featuremap('Enhancement Weights', feature[3].flipud())
    else:
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # sns.heatmap(
        #     torch.sum(make_grid(feature[0].detach(), nrow=int(feature[0].size(0) ** 0.5), padding=2), dim=0).cpu().numpy(),
        #     annot=False, fmt='g', ax=ax)
        # ax.set_title('CNNVision', fontsize=10)
        # fig.savefig(os.path.join(args.save, 'CNNVision.jpg'), dpi=fig.dpi)
        # plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # sns.heatmap(make_grid(feature[1].detach(), nrow=int(feature[1].size(0) ** 0.5), padding=2)[0].cpu().numpy(), annot=False,
        #             fmt='g', ax=ax)
        # ax.set_title('Attention Maps Similarity', fontsize=10)
        # fig.savefig(os.path.join(args.save, 'AttMapSimilarity.jpg'), dpi=fig.dpi)
        # plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        # visweight = model.visweight
        feat, visweight = model.get_visualization()
        sns.heatmap(visweight.detach().cpu().numpy(), annot=False, fmt='g', ax=ax)
        ax.set_title('Enhancement Weights', fontsize=10)
        fig.savefig(os.path.join(args.save, 'EnhancementWeights.jpg'), dpi=fig.dpi)
        plt.close()

    #------------------------------------------
    # Spatial feature visualization
    #------------------------------------------
    headmap = feat.detach().cpu().numpy()
    headmap = np.mean(headmap, axis=1)
    headmap /= np.max(headmap)
    headmap = torch.from_numpy(headmap)

    b, c, t, h, w = inputs.shape
    inputs = inputs.permute(2, 0, 1, 3, 4) #.view(t, b, c, h, w)
    imgs = []
    for img in inputs:
        img = make_grid(img[:16], nrow=4, padding=2).unsqueeze(0)
        imgs.append(img)
    imgs = torch.cat(imgs)
    
    b, t, h, w = headmap.shape
    headmap = headmap.permute(1, 0, 2, 3).unsqueeze(2) # .view(t, b, 1, h, w)
    heatmaps = []
    for heat in headmap:
        heat = make_grid(heat[:16], nrow=4, padding=2)[0].unsqueeze(0)
        heatmaps.append(heat)
    heatmaps = torch.cat(heatmaps)
    
    # feat = model.feat
    # headmap = feat[0,:].detach().cpu().numpy()
    # headmap = np.mean(headmap, axis=0)
    # headmap /= np.max(headmap)  # torch.Size([64, 7, 7])
    # headmap = torch.from_numpy(headmap)
    # img = inputs[0]

    result_gif, result = [], []
    for cam, mg in zip(heatmaps.unsqueeze(1), imgs.permute(0,2,3,1)):
        # cam = torch.argmax(weight_softmax[0]).detach().cpu().dot(cam)
        cam = cv2.resize(cam.squeeze().cpu().numpy(), (mg.shape[0]//2, mg.shape[1]//2))
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        mg = np.uint8(mg.cpu().numpy() * 128 + 127.5)
        mg = cv2.resize(mg, (mg.shape[0]//2, mg.shape[1]//2))
        superimposed_img = cv2.addWeighted(mg, 0.4, cam, 0.6, 0)
        result_gif.append(Image.fromarray(superimposed_img))
        result.append(torch.from_numpy(superimposed_img).unsqueeze(0))
    superimposed_imgs = torch.cat(result).permute(0, 3, 1, 2)
    # save_image(superimposed_imgs, os.path.join(args.save, 'CAM-Features.png'), nrow=int(superimposed_imgs.size(0) ** 0.5), padding=2).permute(1,2,0)
    superimposed_imgs = make_grid(superimposed_imgs, nrow=int(superimposed_imgs.size(0) ** 0.5), padding=2).permute(1,2,0)
    cv2.imwrite(os.path.join(args.save, 'CAM-Features.png'), superimposed_imgs.numpy())
    # save augmentad frames as gif 
    result_gif[0].save(os.path.join(args.save, 'CAM-Features.gif'), save_all=True, append_images=result_gif[1:], duration=100, loop=0)

    if args.eval_only and args.visdom['enable']:
        MHAS_s, MHAS_m, MHAS_l = feature[-2]
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
        sns.heatmap(mhas_s.squeeze(), annot=False, fmt='g', ax=ax, yticklabels=False)
        ax.set_title('\nMHSA Small', fontsize=10)

        ax = fig.add_subplot(132)
        sns.heatmap(mhas_m.squeeze(), annot=False, fmt='g', ax=ax, yticklabels=False)
        ax.set_title('\nMHSA Medium', fontsize=10)

        ax = fig.add_subplot(133)
        sns.heatmap(mhas_l.squeeze(), annot=False, fmt='g', ax=ax, yticklabels=False)
        ax.set_title('\nMHSA Large', fontsize=10)
        plt.suptitle('{}'.format(v_path[0].split('/')[-1]), fontsize=20)
        fig.savefig('demo/{}-MHAS.jpg'.format(args.save.split('/')[-1]), dpi=fig.dpi)
        plt.close()

def feature_embedding(x, target, embedding_dict):
    temp_out, target_out = x

    if temp_out is None:
        x_gather = concat_all_gather(target_out)
        target_gather = concat_all_gather(target.cuda())
        for name, v in zip(target_gather, x_gather):
            embedding_dict[name] = v
    else:
        class_embedding = torch.cat([target_out[i][-1].unsqueeze(-1) for i in range(len(target_out))], dim=-1).mean(-1)
        embedding_gather = concat_all_gather(class_embedding)
        target_gather = concat_all_gather(target.cuda())

        # embedding_dict = OrderedDict()
        for name, v in zip(target_gather, embedding_gather):
            embedding_dict[name] = v
