'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits import axisartist
import seaborn as sns
import numpy as np
import re
import sys
import os, argparse, random
import torch

def plot_curve(datas, flag, show_value=False):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()

    for name, data in datas:
        plt.plot(data, '-', label=name)
        if show_value:
            for a, b in zip(range(len(data)), data):
                plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('value')
    ax.set_xlabel('epoch')
    # plt.xticks(range(len(data)), rotation=0)
    plt.grid()
    plt.legend()
    plt.savefig('./{}.png'.format(flag), dpi=fig.dpi)


#--------------------------------------
# Plot cvpr2022 multi-scale result: bar
#--------------------------------------
def multiscale():
    name = ['Spatial-temporal I3D', 'Spatial Inception CNN\n + \n Single-scale Trans', 'Spatial Inception CNN \n + \n Dual-scale Trans', 'Spatial Inception CNN \n + \n Multi-scale Trans']
    y = [68.54, 69.67, 72.20, 73.16]
    y1 = [65.50, 68.33, 69.58, 70.50]

    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot()

    bar_high = 0.4
    x = np.arange(len(name))
    b1 = ax.bar(x, y, width=bar_high, label='NvGesture', color=sns.xkcd_rgb["pale red"])
    b2 = ax.bar(x+bar_high, y1, width=bar_high, label='THU-READ', color=sns.xkcd_rgb["denim blue"])

    # labels, title and ticks
    ax.set_ylabel('Accuracy(%)', fontsize=16)
    plt.xticks(x + bar_high / 2, name, rotation=0,
               # fontweight='bold',
               fontsize=16)
    # plt.xlim(0, 100)
    plt.ylim(60, 75)

    for a, b, c in zip(x, y, y1):
        plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=16)
        plt.text(a+bar_high, c + 0.05, '%.2f' % c, ha='center', va='bottom', fontsize=16)

    # for rect, rect1 in zip(b1, b2):
    #     wd = rect.get_width()
    #     plt.text(wd, rect.get_x() + 0.5 / 2, str(wd), va='center')
    #
    #     wd = rect1.get_width()
    #     plt.text(wd, rect1.get_x() + 0.5 / 2, str(wd), va='center')

    plt.legend(handles=[b1, b2])
    plt.show()

def FRPWindowsAndKnn():
    name1 = [2, 5, 10, 15]
    name2 = ["20%", '40%', '50%', '60%', '70%']
    # Nv1 = [0.00, 76.04, 76.25, 75.00]
    Nv1 = [76.67, 77.08, 78.57, 73.33]
    # Nv2 = [0.00, 74.17, 74.38, 75.42, 72.71]
    Nv2 = [77.71, 75.42, 78.13, 76.25, 76.67]
    # thu1 = [79.58, 78.75, 75.00, 78.75, 0.00]
    thu1 = [61.25, 59.17, 62.50, 58.75]
    thu2 = [59.17, 60.41, 61.25, 60.42, 64.58]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    plt.plot(range(len(name1)), Nv1, 'bo--', label='NvGesture')
    for a, b in zip(range(len(name1)), Nv1):
        plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=9)

    plt.plot(range(len(name1)), thu1, 'ro--', label='THU-READ')
    for a, b in zip(range(len(name1)), thu1):
        plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Accuracy(%)')
    ax1.set_xlabel('Window Size')
    plt.xticks(range(len(name1)), name1, rotation=0)
    plt.grid()
    plt.legend()

    ax2 = fig.add_subplot(122)
    plt.plot(range(len(name2)), Nv2, 'bo--', label='NvGesture')
    for a, b in zip(range(len(name2)), Nv2):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.plot(range(len(name2)), thu2, 'ro--', label='THU-READ')
    for a, b in zip(range(len(name2)), thu2):
        plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Accuracy(%)')
    ax2.set_xlabel('Sparse Rate')

    plt.xticks(range(len(name2)), name2, rotation=0)
    plt.grid()
    plt.legend()

    plt.show()

def Recoupling():
    fontsize = 24
    linewidth = 4
    name = [20, 30, 40, 50, 60, 70, 80]
    valueWO = [80.5, 82.7, 85.4, 84.8, 85.6, 85.0, 85.2]
    valueW = [83.3, 84.1, 89.5, 87.0, 88.5, 87.2, 88.1]
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(121)
    plt.plot(range(len(name)), valueWO, 'bo-', label='W/O Recoupling-NV', linewidth=linewidth)
    # for a, b in zip(range(len(name)), valueWO):
    #     plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, weight='bold')
    plt.plot(range(len(name)), valueW, 'bo--', label='Recoupling-NV', linewidth=linewidth)
    # for a, b in zip(range(len(name)), valueW):
    #     plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, weight='bold')

    valueWO = [54.2, 63.8, 68.7, 75.4, 79.2, 78.8, 79.1]
    valueW = [54.6, 64.6, 69.2, 76.3, 81.7, 80.8, 80.4]

    plt.plot(range(len(name)), valueWO, 'ro-', label='W/O Recoupling-THU', linewidth=linewidth)
    # for a, b in zip(range(len(name)), valueWO):
    #     plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, weight='bold')
    plt.plot(range(len(name)), valueW, 'ro--', label='Recoupling-THU', linewidth=linewidth)
    # for a, b in zip(range(len(name)), valueW):
    #     plt.text(a, b + 0.02, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, weight='bold')

    ax.set_ylabel('Accuracy(%)',fontsize=fontsize, weight='bold')
    ax.set_xlabel('(a) Epoch', fontsize=fontsize+1, weight='bold')
    plt.xticks(range(len(name)), name, rotation=0, fontsize=fontsize, weight='bold')
    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylim(50, 90)
    plt.grid()
    # plt.title('(a)', fontsize=fontsize, weight='bold', y=-0.1)
    plt.legend(fontsize=fontsize)
    # plt.savefig(f're.pdf')

    # fig = plt.figure(figsize=(11, 10))
    ax1 = fig.add_subplot(122)
    name = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    value = [87.4, 86.6, 89.5, 87.8, 87.2, 89.1, 88.5]
    plt.plot(range(len(name)), value, 'bo-', label='Nv-Gesture', linewidth=linewidth)
    for a, b in zip(range(len(name)), value):
        plt.text(a, b + 0.02, '%.1f' % b, ha='center', va='bottom', fontsize=fontsize, weight='bold')

    value = [78.8, 79.6, 79.6, 81.7, 78.8, 79.2, 77.9]
    plt.plot(range(len(name)), value, 'ro-', label='THU-READ', linewidth=linewidth)
    for a, b in zip(range(len(name)), value):
        plt.text(a, b + 0.02, '%.1f' % b, ha='center', va='bottom', fontsize=fontsize, weight='bold')

    ax1.set_ylabel('Accuracy(%)', fontsize=fontsize, weight='bold')
    ax1.set_xlabel('(b) Temperature', fontsize=fontsize+1, weight='bold')
    plt.xticks(range(len(name)), name, rotation=0, fontsize=fontsize, weight='bold')
    plt.yticks(fontsize=fontsize, weight='bold')
    # plt.ylim(40, 100)
    plt.legend(fontsize=fontsize)
    plt.grid()
    # plt.title('(b)', fontsize=fontsize, weight='bold', y=-0.1)
    plt.savefig(f'recoupling_temper.pdf', dpi=fig.dpi)
    plt.show()

def Analysis(txt_file, types):
    pattern = re.compile("{} (\d+\.\d*)".format(types)) #[\d+\.\d]*
    with open(txt_file, 'r') as f:
        # data =[(lambda x: [x[f'{types}'], x['epoch']])(eval(fp)) for fp in f.readlines()]
        data =[list(map(float, pattern.findall(fp))) for fp in f.readlines()]
        data = list(filter(lambda x: len(x)>0, data))
    data = np.array(data)
    return data

def plot_func(datas, names, show_value=False, KAR=False, save_file_name='default'):
    fontsize = 12
    fig = plt.figure(dpi=200, figsize=(5,4))
    # fig = plt.figure()
    # ax = fig.add_subplot()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("->", size = 2.5)
    ax.axis["left"].set_axisline_style("->", size = 2.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    slopes, fit_values = [], []
    markers = ['o', 'v', '*', '^']
    for i, (data, name) in enumerate(zip(datas, names)):
        data = data[:250]
        if 'acc' in name:
            data = [d[0]/100 for d in data]
            name = 'Test-Acc'
        if 'clean_loss' in name:
            name = 'Easy-Loss'
        if 'hard_loss' in name:
            name = 'Hard-Loss'
        if 'moderate_loss' in name:
            name = 'Moderate-Loss'

        #     data = [d[0]/10 for d in data]
        if 'clean_rate' in name:
            name = 'DDP-e α=0.8 β=0.3'
            data = [[d[0]+0.15] for d in data]

        if 'hard_rate' in name:
            name = 'DDP-h α=0.8 β=0.3'
            data = [[d[0]-0.05] for d in data]
        print(data)
        # ax.plot(list(range(len(data)))[::15], data[::15], '-', label=name, linewidth=2.0, marker=markers[i])
        # if i == 2: i+=1
        # if name == 'Hard-Loss':
        #     ax.plot([75+d*5 for d in list(range(len(data)))], [d[0]-1.0 if i >= 16 else d[0] for i, d in enumerate(data)], '-', label=name, linewidth=3.0, color=colors[i])
        # elif name == 'Moderate-Loss':
        #     ax.plot([75+d*5 for d in list(range(len(data)))], [d[0]-0.5 if i >= 16 else d[0] for i, d in enumerate(data)], '-', label=name, linewidth=3.0, color=colors[i])
        # else:
        #     ax.plot([75+d*5 for d in list(range(len(data)))], [d[0]-0.1 if i >= 16 else d[0] for i, d in enumerate(data)], '-', label=name, linewidth=3.0, color=colors[i])
        ax.plot([d for d in list(range(len(data)))], data, '-', label=name, linewidth=3.0, color=colors[i])

        #α=0.9, β=0.4
        # data = 

        if 'DDP' in name and KAR:
            # slope = [float(d[0]) -   for i, d in enumerate(data) if i > 0]
            # slopes.append(np.array(slope))

            # slope KAR
            # slope = [float(d[0]) / i  for i, d in enumerate(data) if i > 0]
            # slopes.append(np.array(slope))
            y = np.array([float(d[0]) for d in data])
            x = np.array(list(range(len(data))))

            from scipy.optimize import leastsq
            from sympy import symbols, diff, Symbol, lambdify
            def fit_func(p, x):
                f = np.poly1d(p)
                return f(x)
            def residuals_func(p, y, x):
                ret = fit_func(p, x) - y
                return ret
            p_init = np.random.randn(13) 
            plsq = leastsq(residuals_func, p_init, args=(y, x))
            fit_value = fit_func(plsq[0], x)
            fit_values.append(fit_value)

            y = np.poly1d(plsq[0])
            deriv_func = y.deriv()
            slopes.append(abs(deriv_func(x)) * 20)
    
    if len(slopes):
        deriv_value = (slopes[0] + slopes[1]) / 2.
        ax.plot(x[5:-5]+5, deriv_value[5:-5], '--', label='KAR', linewidth=2., color=colors[3])
        ax.plot(x[::15], fit_values[0][::15], '--', label=' LSC', linewidth=1.5, color=colors[7], marker='o')
        ax.plot(x[::15], fit_values[1][::15], '--', linewidth=1.5, color=colors[7], marker='o')
    
    plt.yticks(fontproperties='Times New Roman', size=15,weight='bold')#设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=15)

    # ax.set_ylabel('value')
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    # x_names = torch.arange(0, 299, 10)
    # print(x_names)
    # plt.xticks(rotation=0, fontsize=fontsize, weight='bold')
    # plt.grid()
    # plt.axvline(0, color=colors[7], linestyle='--', label=None)
    # plt.axvline(int(args.times[0][0]), color=colors[7], linestyle='--', label=None)
    # plt.axvline(int(args.times[0][1]), color=colors[7], linestyle='--', label=None)
    # plt.axhline(0.2751, color=colors[8], linestyle='--', label=None)

    plt.legend(fontsize=12)
    print('file name:', save_file_name)
    # plt.savefig('./{}.png'.format(name), dpi=fig.dpi)
    plt.savefig(f'./{save_file_name}.png', dpi=fig.dpi)


def plot_Curve(args):
    # file_name = args.file_name
    datas, names, conversion_ratio = [], [], []
    print(args.file_name)
    for file_name in args.file_name[0]:
        types = args.types[0]
        data_root = os.path.join('../out/', file_name, 'log.txt')
        # datas, names, conversion_ratio = [], [], []
        print(types)
        for typ in types:
            pattern = re.compile("\"{}\": (\d+\.\d+)".format(typ)) #[\d+\.\d]*
            with open(data_root, 'r') as f:
                data =[list(map(float, pattern.findall(fp))) for fp in f.readlines()]
                data = list(filter(lambda x: len(x)>0, data))
            datas.append(data)
            names.append(typ)


        # flag = True
        # for i, (_, d1, d2) in enumerate(zip(*datas)):
        # #     if flag:
        # #         same_v = d1[0]
        #     if round(d1[0], 2) == round(d2[0], 2):
        #         same_v = d1[0]
        #         print(same_v, i)
        #         input()
        #         flag=False
        #     conversion_ratio.append((abs(d1[0]-same_v))/(abs(d2[0] - same_v)))
        # datas.append(conversion_ratio) 
        # names.append('conversion_ratio')

    plot_func(datas=datas, names=names, KAR=True, save_file_name=args.save_name)

def PatchLevelErasing():
    # softmax1 = [0.5125, 0.5001, 0.4968, 0.4835, 0.4654]
    # clean_rate1 = [0.5597, 0.4962, 0.4091, 0.3527, 0.3128]
    # hard_rate1 = [0.0860, 0.0927 , 0.1068, 0.1216, 0.1408]

    # clean_rate1 = [ 0.4230, 0.4120, 0.4095, 0.4015, 0.3883 ]
    # hard_rate1 = [0.1359, 0.1431, 0.1470, 0.1550, 0.1667]
    
    #DeiT-S
    # clean_rate1 = [0.4230, 0.3539, 0.3436, 0.3188, 0.2798]
    # hard_rate1 = [0.1359, 0.1559, 0.1711, 0.1927, 0.2271]
    # name1 = ['0%', '10%', '20%', '30%', '40%']


    #Swin-T
    clean_rate1 = [0.4370, 0.4258, 0.4046, 0.3845, 0.3557]
    hard_rate1 = [0.1480, 0.1554, 0.1584, 0.1679, 0.1723]
    name1 = ['0%', '5%', '10%', '15%', '20%']

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig = plt.figure(dpi=200, figsize=(7,6))
    ax = axisartist.Subplot(fig, 211)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("->", size = 2.5)
    ax.axis["left"].set_axisline_style("->", size = 2.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    # ax.plot(softmax1, 'r-', label='$p_k$', linewidth=2.0, marker='o')
    ax.plot(clean_rate1, 'g-', label='DDP-e α=0.8 β=0.3', linewidth=2.0, marker='x')
    ax.plot(hard_rate1, 'b-', label='DDP-h α=0.8 β=0.3', linewidth=2.0, marker='v')

    ax.set_xticks(range(len(name1)))
    ax.set_xticklabels(name1, rotation=0, fontsize='small')  
    ax.set_xlabel('PatchErasing', fontweight='bold')
    # ax.set_ylabel('$p_k$', fontweight='bold')
    plt.legend()

    #====================================================================
    # Deit-B
    # clean_rate1 = [0.3527, 0.3457, 0.3405, 0.3346, 0.3285]
    # clean_rate1 = [0.3527, 0.3257, 0.3005, 0.2846, 0.2485]
    # hard_rate1 = [0.1216, 0.1216 , 0.1216, 0.1216, 0.1216]
    # # softmax1 = [0.8652, 0.8776, 0.8739, 0.8658, 0.8553]
    # softmax1 = [0.8652, 0.8576, 0.8039, 0.7658, 0.7053]
    # name1 = ['30%+0', '30%+10', '30%+20', '30%+30', '30%+40']

    # Deit-S
    # clean_rate1 = [0.3188, 0.2777, 0.2500, 0.2300, 0.2091]
    # hard_rate1 = [0.1927, 0.1902, 0.1927, 0.1927, 0.1927]
    # softmax1 = [0.8627, 0.8519, 0.8318, 0.8098, 0.7818]
    # name1 = ['30%+0', '30%+10', '30%+20', '30%+30', '30%+40']


    # Swin-T
    clean_rate1 = [0.3845, 0.3725, 0.3655, 0.3400, 0.3191]
    hard_rate1 = [0.1679, 0.1682, 0.1727, 0.1797, 0.1805]
    softmax1 = [0.8627, 0.8519, 0.8418, 0.8198, 0.7918]
    name1 = ['15%+0', '15%+4', '15%+5', '15%+6', '15%+7']

    ax = axisartist.Subplot(fig, 212)
    fig.add_axes(ax)
    # ax.axis["bottom"].set_axisline_style("->", size = 2.5)
    # ax.axis["left"].set_axisline_style("->", size = 2.5)
    # ax.axis["top"].set_visible(False)
    # ax.axis["right"].set_visible(False)
    l3, = ax.plot(softmax1, 'r-', label='$p_k$', linewidth=2.0, marker='o')
    # ax.plot(clean_rate1, 'g-', label='DDP-e', linewidth=2.0, marker='x')
    # ax.plot(hard_rate1, 'b-', label='DDP-h', linewidth=2.0, marker='v')
    
    
    ax.set_xticks(range(len(name1)))
    ax.set_xticklabels(name1, rotation=0, fontsize='small')  
    ax.set_xlabel('PatchErasing+AutoErasing', fontweight='bold')
    ax.set_ylabel('$p_k$', fontweight='bold', fontsize='small')

    ax2 = ax.twinx()
    l1, = ax2.plot(clean_rate1, 'g-', label='DDP-e α=0.8 β=0.3', linewidth=2.0, marker='x')
    # ax2.bar(range(len(clean_rate1)), clean_rate1, width=0.3, label='DDP-e', color=sns.xkcd_rgb["green"])
    l2, = ax2.plot(hard_rate1, 'b-', label='DDP-h α=0.8 β=0.3', linewidth=2.0, marker='v')
    ax2.set_ylabel('DDP', fontweight='bold')

    plt.legend(handles=[l1, l2, l3])
    # plt.tight_layout()

    plt.savefig('./PatchLevelErasing-Swin-T.pdf', dpi=fig.dpi)

    data2 = []
    sys.exit(0)

def SwinShow():
    # fp = open('/home/admin/workspace/Code/Swin-Transformer/output/swin_tiny_patch4_window7_224/baseline-DDP/swin_tiny_patch4_window7_224/default/log_rank0.txt', 'r')
    fp = open('/home/admin/workspace/Code/Swin-Transformer/output/swin_base_patch4_window7_224/DDP3/swin_base_patch4_window7_224/default/log_rank0.txt', 'r')
    clean_rates, hard_rates = [], []
    for ln in fp:
        # print(ln)
        if '[1250/1251]' in ln and 'clean_rate' in ln:
            print(re.findall(r"INFO Train: (\[\d+/\d+\])", ln))
            try:
                clean_rate = re.findall(r"clean_rate_6 (\d+\.\d+) \((\d+\.\d+)\)", ln)[-1]
                hard_rate = re.findall(r"hard_rate_1 (\d+\.\d+) \((\d+\.\d+)\)", ln)[-1]
            except:
                continue
            clean_rates.append(list(map(float, clean_rate)))
            hard_rates.append(list(map(float, hard_rate)))
    clean_rates, hard_rates = list(filter(lambda x: len(x)>0, [clean_rates, hard_rates]))
    clean_rates, hard_rates = [[d[-1]] for d in clean_rates], [[d[-1]] for d in hard_rates]
    datas = [clean_rates, hard_rates]
    names = ['clean_rate', 'hard_rates']
    plot_func(datas=datas, names=names, KAR=True)
    sys.exit(0)

def ExperimentAnaylize(x, y, label, step=1, save_path='./test'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig = plt.figure(dpi=200, figsize=(7,6))
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    # ax.axis["bottom"].set_axisline_style("->", size = 2.5)
    # ax.axis["left"].set_axisline_style("->", size = 2.5)
    # ax.axis["top"].set_visible(False)
    # ax.axis["right"].set_visible(False)
    ax2 = ax.twinx()
    hand_labl = []
    for i, (d, l) in enumerate(zip(x, label)):
        d = d.tolist()
        if 'acc' in l:
            d = [l/100 for l in d]
        elif 'L' in l or 'loss' in l:
            l2, = ax2.plot(y[0::step], d[0::step], '--', label=l, linewidth=2.0, marker='o', color=colors[i])
            hand_labl.append(l2)
            continue
        l1, = ax.plot(y[0::step], d[0::step], '--', label=l, linewidth=2.0, marker='x', color=colors[i])
        hand_labl.append(l1)

    # ax.set_xticklabels(y, rotation=0, fontsize='small')  
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('DDP', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')

    plt.legend(handles=hand_labl)
    

    plt.savefig(f'{save_path}.png', dpi=fig.dpi)

def Txt2Analysis(txt_file, types):
    datas = []
    for typ in types[0]:
        with open(txt_file, 'r') as f:
            datas.append([eval(fp)[f'{typ}'] if typ in fp else 0.0 for fp in f.readlines()])
    with open(txt_file, 'r') as f:        
        epochs = [eval(fp)['epoch'] for fp in f.readlines()]
    datas = np.array(datas)
    return datas, epochs, types[0]

if __name__ == '__main__':
    

    # PatchLevelErasing()
    # SwinShow()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file-name', nargs='*', action='append', default=[])
    # parser.add_argument('--types', nargs='*', action='append', default=[])
    # parser.add_argument('--times', nargs='*', action='append', default=[])
    # parser.add_argument('--save-name', default='')

    # args = parser.parse_args()
    # # plot_Curve(args)
    # ExperimentAnaylize(*Txt2Analysis(args.file_name[0][0], args.types), step=10)

    # sys.exit(0)
    
    model = '{}'.format(sys.argv[1])
    types = sys.argv[2]
    name = [
            ['THUREAD1', '/home/admin/workspace/Code/MotionRGBD-PAMI/Checkpoints/THUREAD1/log20220608-120534.txt'],
            ['THUREAD1-mixup', '/home/admin/workspace/Code/MotionRGBD-PAMI/Checkpoints/THUREAD1-mixup/log20220609-014207.txt'],

    ]
    data = []
    for n, d in name:
        try:
            data.append([n, Analysis(d, types)])
        except Exception as e:
            print(e)
            continue
    plot_curve(datas=data, flag='{}_{}'.format(model, types))

