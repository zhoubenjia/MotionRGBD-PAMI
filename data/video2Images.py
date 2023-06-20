import cv2
from PIL import Image
import numpy as np

import os, glob, re, sys
import argparse
import csv
import random
from tqdm import tqdm
from multiprocessing import Process
import shutil
from multiprocessing import Pool, cpu_count


def video2image(v_p):
    img_path = os.path.join(args.img_path, v_p.split('/')[-1][:-4])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(v_p)
    suc, frame = cap.read()
    frame_count = 0
    while suc:
        h, w, c = frame.shape
        cv2.imwrite('{}/{:0>6d}.jpg'.format(img_path, frame_count), frame)
        frame_count += 1
        suc, frame = cap.read()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', '-vp', default='')
    parser.add_argument('--save-path', '-sp', default='')
    parser.add_argument('--pool-num', '-p', default=1, type=int, help='Number of used multiple processes.')

    args = parser.parse_args()

    v_path = os.path.join(args.video_path, '*/*.mp4')
    videos = glob.glob(v_path)
    assert len(videos) >= 1, f'Please Check Video Path ! There are any videos in the dir {v_path}.'
    print('Total Videos: {}'.format(len(videos)))

    args.img_path = os.path.join(args.save_path, 'Images')
    str = input(f'The video will be decompressed to: {args.img_path}, [y/n]')
    if str == 'n':
        sys.exit(0)
    
    with Pool(args.pool_num) as pool:
        for a in tqdm(pool.imap_unordered(video2image, videos), total=len(videos), desc='Processes'):
            if a is not None:
                pass
    print('Decompressing done'.center(80, '*'))


# class UnsupportedFormat(Exception):
#     def __init__(self, input_type):
#         self.t = input_type

#     def __str__(self):
#         return "不支持'{}'模式的转换，请使用为图片地址(path)、PIL.Image(pil)或OpenCV(cv2)模式".format(self.t)


# class MatteMatting():
#     def __init__(self, original_graph, mask_graph, input_type='cv2'):
#         """
#         将输入的图片经过蒙版转化为透明图构造函数
#         :param original_graph:输入的图片地址、PIL格式、CV2格式
#         :param mask_graph:蒙版的图片地址、PIL格式、CV2格式
#         :param input_type:输入的类型，有path：图片地址、pil：pil类型、cv2类型
#         """
#         if input_type == 'path':
#             self.img1 = cv2.imread(original_graph)
#             self.img2 = cv2.imread(mask_graph)
#         elif input_type == 'pil':
#             self.img1 = self.__image_to_opencv(original_graph)
#             self.img2 = self.__image_to_opencv(mask_graph)
#         elif input_type == 'cv2':
#             self.img1 = original_graph
#             self.img2 = mask_graph
#         else:
#             raise UnsupportedFormat(input_type)

#     @staticmethod
#     def __transparent_back(img):
#         """
#         :param img: 传入图片地址
#         :return: 返回替换白色后的透明图
#         """
#         img = img.convert('RGBA')
#         L, H = img.size
#         color_0 = (255, 255, 255, 255)  # 要替换的颜色
#         for h in range(H):
#             for l in range(L):
#                 dot = (l, h)
#                 color_1 = img.getpixel(dot)
#                 if color_1 == color_0:
#                     color_1 = color_1[:-1] + (0,)
#                     img.putpixel(dot, color_1)
#         return img

#     def save_image(self, path, mask_flip=False):
#         """
#         用于保存透明图
#         :param path: 保存位置
#         :param mask_flip: 蒙版翻转，将蒙版的黑白颜色翻转;True翻转;False不使用翻转
#         """
#         if mask_flip:
#             img2 = cv2.bitwise_not(self.img2)  # 黑白翻转
#         image = cv2.add(self.img1, img2)
#         image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV转换成PIL.Image格式
#         img = self.__transparent_back(image)
#         img.save(path)

#     @staticmethod
#     def __image_to_opencv(image):
#         """
#         PIL.Image转换成OpenCV格式
#         """
#         img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#         return img

# data_root = '/mnt/workspace/Dataset/UCF-101/'
# label_dict = dict([(lambda x: (x[1], int(x[0])-1))(l.strip().split(' ')) for l in open(data_root + 'dataset_splits/lableind.txt').readlines()])
# print(label_dict)

# def split_func(file_list):
#     class_list = []
#     fl = open(file_list).readlines()
#     for d in tqdm(fl):
#         path = d.strip().split()[0][:-4]
#         label = label_dict[path.split('/')[0]]
#         frame_num = len(os.listdir(os.path.join(data_root, 'UCF-101-images', path)))
#         class_list.append([path, str(frame_num), str(label), '\n'])
#     return class_list

# def save_list(file_list, file_name):
#     with open(file_name, 'w') as f:
#         class_list = split_func(file_list)
#         for l in class_list:
#             f.writelines(' '.join(l))

# prot = '@3'
# data_train_split = data_root + f'dataset_splits/{prot}/trainlist.txt'
# data_test_split = data_root + f'dataset_splits/{prot}/testlist.txt'

# train_file_name = data_root + f'dataset_splits/{prot}/train.txt'
# test_file_name = data_root + f'dataset_splits/{prot}/valid.txt'
# save_list(data_train_split, train_file_name)
# save_list(data_test_split, test_file_name)