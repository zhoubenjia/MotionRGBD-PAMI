'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
import pandas as pd
from sklearn import manifold
import numpy as np
import sys
import sklearn
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

M_path = '/mnt/workspace/Code/MotionRGBD-PAMI/Checkpoints/THU-READ4-32-DTNV2-M-simi/'
K_path = '/mnt/workspace/Code/MotionRGBD-PAMI/Checkpoints/THU-READ4-32-DTNV2-K-simi-cross/'

def normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

data = []
for i in range(0, 100, 5): 
    M_checkpoint = M_path + f'/feature-M-epoch{i}.pth'
    M_features = torch.load(M_checkpoint, map_location='cpu')
    K_checkpoint = K_path + f'/feature-K-epoch{i}.pth'
    K_features = torch.load(K_checkpoint, map_location='cpu')
    
    simlitary = []
    for (km, vm), (kd, vd) in zip(M_features.items(), K_features.items()):
        assert km == kd
        # pca_data = pd.DataFrame(vm.cpu().numpy())
        # vm = torch.tensor(tsne.fit_transform(pca_data))
        # pca_data = pd.DataFrame(vd.cpu().numpy())
        # vd = torch.tensor(tsne.fit_transform(pca_data))
        
        # vm, vd = normalization(vm), normalization(vd) #F.normalize(vm, p = 2, dim=-1), F.normalize(vd, p = 2, dim=-1) #
        simil = F.pairwise_distance(vm.unsqueeze(0), vd.unsqueeze(0), p=2)
        # simil = torch.cosine_similarity(vm, vd, dim=-1)
        # simil = torch.tensor(pdist(np.vstack([vm.numpy(),vd.numpy()]),'seuclidean')[0])
        # simil = vm * vd
        
        
        simlitary.append(simil.unsqueeze(0))
    simi_value = torch.cat(simlitary).mean()
    data.append(float(simi_value))
    
    
M_embed = torch.cat([F.normalize(e.unsqueeze(0), p = 2, dim=-1) for e in M_features.values()])
K_embed = torch.cat([F.normalize(e.unsqueeze(0), p = 2, dim=-1) for e in K_features.values()])
embed = torch.cat((M_embed, K_embed))
label_embed = torch.cat((torch.ones(M_embed.shape[0]), torch.ones(M_embed.shape[0])+1))

tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
embed = pd.DataFrame(embed.cpu().numpy())
pca_embed = tsne.fit_transform(embed)
embed.insert(0, 'label', pd.DataFrame(label_embed.cpu().numpy()))
print(pca_embed.shape)

fig, ax = plt.subplots()
scatter = ax.scatter(pca_embed[:, 0], pca_embed[:, 1], c=embed['label'], s=25, cmap='rainbow', 
alpha=0.8, edgecolors='none')
plt.savefig("./"+'cluster.png', dpi=120, bbox_inches='tight')