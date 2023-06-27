import numpy as np
import torch
import random

def Vmixup(x, lam):
    x_flipped = x.flip(0).mul_(1. - lam)
    x.mul_(lam).add_(x_flipped)

def ShuffleMix(x, lam):
    x_flipped = x.flip(0)
    replace_idx = random.sample(range(0, x.size(2)), round((1. - lam)*x.size(2)))
    x[:, :, replace_idx, :, :] = x_flipped[:, :, replace_idx, :, :]

def ShuffleMix_v1(x, lam):
    x_flipped = x.flip(0)
    length = x.size(2)
    start = random.sample([0, 1], 1)[0]
    a = torch.arange(start, length, step=2)
    v_len = int(length*(1. - lam))
    replace_num = min(v_len, length-v_len)
    if len(a)-replace_num:
        b = random.sample(range(0, len(a)-replace_num), 1)[0]
    else:
        b = 0
    replace_idx = a[b:b+replace_num]
    if v_len <= length-v_len:
        x[:, :, replace_idx, :, :] = x_flipped[:, :, replace_idx, :, :]
    else:
        x_flipped[:, :, replace_idx, :, :] = x[:, :, replace_idx, :, :]
        replace_idx = torch.arange(0, x.size(2), step=1)
        x[:, :, replace_idx, :, :] = x_flipped[:, :, replace_idx, :, :]

def ShuffleMix_v2(x, lam):
    x_flipped = x.flip(0)
    # replace_idx = random.sample(range(0, x.size(2)), int((1. - lam)*x.size(2)))
    length = max(1, int((1. - lam)*x.size(2)))
    # uni_idx = uniform_sampling(x.size(2), length, random=True)

    if x.size(2) != length:
        start = random.sample(range(0, x.size(2) - length), 1)[0]
        replace_idx = torch.arange(start, start+length, step=1)
        # x[:, :, replace_idx, :, :] = x[:, :, replace_idx, :, :].mul_(lam).add_(x_flipped[:, :, uni_idx, :, :].mul_(1. - lam))
        # x[:, :, replace_idx, :, :] = x_flipped[:, :, replace_idx, :, :]
        x[:, :, -len(replace_idx):] = x_flipped[:, :, replace_idx, :, :]
    else:
        x = x_flipped

def ShuffleMix_v3(x, lam):
    x_flipped = x.flip(0)
    length = int((1. - lam)*x_flipped.size(2))
    # if length:
    #     x[:, :, -length:, :, :] = x_flipped[:, :, :length, :, :]

    uni_idx = uniform_sampling(x_flipped.size(2), length, random=True)
    x1 = x_flipped[:, :, uni_idx, :, :]

    length = x.size(2) - length
    uni_idx = uniform_sampling(x.size(2), length, random=True)
    x2 = x[:, :, uni_idx, :, :]

    x_cat = torch.cat((x2, x1), dim=2)
    replace_idx = torch.arange(0, x.size(2), step=1)
    x[:, :, replace_idx, :, :] = x_cat[:, :, replace_idx, :, :]
    
    # uni_idx = uniform_sampling(x_flipped.size(2), length, random=True)
    # x1 = x_flipped[:, :, uni_idx, :, :]

    # length = x.size(2) - length
    # uni_idx = uniform_sampling(x.size(2), length, random=True)
    # x2 = x[:, :, uni_idx, :, :]

    # start = random.sample([0, length-1], 1)[0]
    # if start == 0:
    #     x_cat = torch.cat((x1, x2), dim=2)
    # else:
    #     x_cat = torch.cat((x2, x1), dim=2)
    # # x_cat = torch.cat((x2, x1), dim=2)
    # assert x.size(2) == x_flipped.size(2), f'x size {x.size(2)} must match with raw size {x_flipped.size(2)}'
    # replace_idx = torch.arange(0, x.size(2), step=1)
    # x[:, :, replace_idx, :, :] = x_cat[:, :, replace_idx, :, :]

def Mixup_ShuffleMix(x, lam):
    x_flipped = x.flip(0).mul_(1. - lam)
    x.mul_(lam).add_(x_flipped)
    replace_idx = random.sample(range(0, x.size(2)), int((1. - lam)*x.size(2)))
    x[:, :, replace_idx, :, :] = x_flipped[:, :, replace_idx, :, :]

def TempMix(x, lam):
    lam = np.random.beta(lam, lam, size=x.size(2))
    lam = torch.from_numpy(lam)[None, None, :, None, None].cuda()
    x_flipped = x.flip(0).mul_(1. - lam)
    x.mul_(lam).add_(x_flipped)

    return lam.squeeze()

def ShuffleMix_plus(x, lam, smprob):
    # x: torch.Size([16, 3, 16, 224, 224])
    lam = torch.tensor(lam).view(-1).expand(x.size(2)).clone().cuda()
    lam_flipped = 1.0 - lam

    replace_idx = random.sample(range(0, x.size(2)), round(smprob*x.size(2)))
    lam[replace_idx] = lam_flipped[replace_idx]
    lam_flipped = 1.0 - lam

    x_flipped = x.flip(0).mul_(lam_flipped.view(1, 1, -1, 1, 1))
    x.mul_(lam.view(1, 1, -1, 1, 1)).add_(x_flipped)

    return lam.mean()

def MixIntra(x, lam, target, replace_prob):
    # gather from all gpus
    batch_size_this = x.shape[0]
    # x_gather = concat_all_gather(x)
    # target_gather = concat_all_gather(target)

    # batch_size_all = x_gather.shape[0]
    # num_gpus = batch_size_all // batch_size_this

    labes = np.unique(target.cpu().numpy())
    label_dict = dict([(t, []) for t in labes])
    for idx, t in enumerate(target.tolist()):
        label_dict[t].append(idx)
    indx_list = [random.choice(label_dict[t]) for t in target.tolist()]

    x_intra = x[indx_list]
    replace_idx = random.sample(range(0, x.size(2)), round(replace_prob*x.size(2)))
    x[:, :, replace_idx, :, :] = x_intra[:, :, replace_idx, :, :]

    # x_flipped = x_gather.flip(0).mul_(1. - lam)
    # x_gather.mul_(lam).add_(x_flipped)
    
    # random shuffle index
    # idx_shuffle = torch.arange(batch_size_all).cuda()

    # # broadcast tensor to all gpus
    # torch.distributed.broadcast(idx_shuffle, src=0)
    # # shuffled index for this gpu
    # gpu_idx = torch.distributed.get_rank()
    # idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    # x[torch.arange(batch_size_this)] = x_gather[idx_this]
    return 0.9
