'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

NTU-RGBD Data preprocessing function
'''

import cv2
from PIL import Image
import numpy as np

import os, glob, re
import argparse
import csv
import random
from tqdm import tqdm
from multiprocessing import Process
import shutil
from multiprocessing import Pool, cpu_count

def resize_pos(center,src_size,tar_size):
    x, y = center
    w1=src_size[1]
    h1=src_size[0]
    w=tar_size[1]
    h=tar_size[0]

    y1 = int((h / h1) * y)
    x1 = int((w / w1) * x)
    return (x1, y1) 

def video2image_with_mask(v_p):
    m_path='nturgb+d_depth_masked/'
    img_path = os.path.join('NTU-RGBD-images', v_p[:-4].split('/')[-1])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(v_p)
    suc, frame = cap.read()
    frame_count = 1
    while suc:
        # frame resolution: [1920, 1080]
        mask_path = os.path.join(m_path, v_p[:-8].split('/')[-1], 'MDepth-%08d.png'%frame_count)
        assert os.path.isfile(mask_path), FileNotFoundError('[error], file not found.')
        mask = cv2.imread(mask_path)
        mask = mask*255
        w, h, c = mask.shape
        h2, w2, _ = frame.shape
        ori = frame
        frame = cv2.resize(frame, (h, w))
        h1, w1, _ = frame.shape

        # image = cv2.add(frame, mask)

        # find contour
        mask = cv2.erode(mask, np.ones((3, 3),np.uint8))
        mask = cv2.dilate(mask ,np.ones((10, 10),np.uint8))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find Max Maxtri
        Idx = []
        for i in range(len(contours)):
            Area = cv2.contourArea(contours[i])
            if Area > 500:
                Idx.append(i)
        # max_idx = np.argmax(area)

        centers = []
        for i in Idx:
            rect = cv2.minAreaRect(contours[i])
            center, (h, w), degree = rect
            centers.append(center)

        finall_center = np.int0(np.array(centers))
        c_x = min(finall_center[:, 0])
        c_y = min(finall_center[:, 1])

        center = (c_x, c_y)
        # finall_center = finall_center.sum(0)/len(finall_center)

        # rect = cv2.minAreaRect(contours[max_idx])
        # center, (h, w), degree = rect
        # center = tuple(np.int0(finall_center))
        center_new = resize_pos(center, (h1, w1), (h2, w2))

        #-----------------------------------
        # Image Crop
        #-----------------------------------
        # ori = cv2.circle(ori, center_new, 2, (0, 0, 255), 2)
        crop_y, crop_x = h2//2, w2//2
        # print(crop_x, crop_y)
        left = center_new[0] - crop_x//2 if center_new[0] - crop_x//2 > 0 else 0
        top = center_new[1] - crop_y//2 if center_new[1] - crop_y//2 > 0 else 0
        # ori = cv2.circle(ori, (left, top), 2, (0, 0, 255), 2)
        # cv2.imwrite('demo/ori.png', ori)
        crop_w = left + crop_x if left + crop_x < w2 else w2
        crop_h = top + crop_y if top + crop_y < h2 else h2
        rect = (left, top, crop_w, crop_h)
        image = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
        image = image.crop(rect)
        image.save('{}/{:0>6d}.jpg'.format(img_path, frame_count))

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # drawImage = frame.copy()
        # drawImage = cv2.drawContours(drawImage, [box], 0, (255, 0, 0), -1)  # draw one contour
        # cv2.imwrite('demo/drawImage.png', drawImage)
        # frame = cv2.circle(frame, center, 2, (0, 255, 255), 2)
        # cv2.imwrite('demo/Image.png', frame)
        # cv2.imwrite('demo/mask.png', mask)
        # ori = cv2.circle(ori, center_new, 2, (0, 0, 255), 2)
        # cv2.imwrite('demo/ORI.png', ori)
        # cv2.imwrite('demo/maskImage.png', image)

        # cv2.imwrite('{}/{:0>6d}.jpg'.format(img_path, frame_count), frame)
        frame_count += 1
        suc, frame = cap.read()
    cap.release()

def video2image(v_p):
    img_path = v_p[:-4].replace('NTU-RGBD', 'NTU-RGBD-images')
    os.makedirs(img_path, exist_ok=True)

    cap = cv2.VideoCapture(v_p)
    suc, frame = cap.read()
    frame_count = 0
    while suc:
        h, w, c = frame.shape
        cv2.imwrite('{}/{:0>6d}.jpg'.format(img_path, frame_count), frame)
        frame_count += 1
        suc, frame = cap.read()
    cap.release()

def GeneratLabel(v_p):
    path = v_p[:-4].split('/')[-1]
    cap = cv2.VideoCapture(v_p)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label = int(v_p.split('A')[-1][:3])-1
    txt = ' '.join(map(str, [path, frame_count, label, '\n']))
    if args.proto == '@CV':
        if 'C001' in v_p:
            with open(args.validTXT, 'a') as vf:
                vf.writelines(txt)
        else:
            with open(args.trainTXT, 'a') as tf:
                tf.writelines(txt)
    elif args.proto == '@CS':
        pattern = re.findall(r'P\d+', v_p) 
        if int(pattern[0][1:]) in [1, 2, 4, 5, 8, 9, 13, 14, 15,16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]:
            with open(args.trainTXT, 'a') as tf:
                tf.writelines(txt)
        else:
            with open(args.validTXT, 'a') as vf:
                vf.writelines(txt)

def ResizeImage(v_p):
    img_path = v_p[:-4].replace('NTU-RGBD', 'NTU-RGBD-images')
    save_path = img_path.replace('NTU-RGBD-images', 'NTU-RGBD-resized-images')
    os.makedirs(save_path, exist_ok=True)

    for img in os.listdir(img_path):
        im_path = os.path.join(img_path, img)
        image = cv2.imread(im_path)
        image = cv2.resize(image, (320, 240))
        cv2.imwrite(os.path.join(save_path, img), image)

#---------------------------------------------
# Generate label .txt
#---------------------------------------------
def generate_label(Video_paths, test_first=True):
    trainTXT = os.path.join(args.data_root, 'dataset_splits', args.proto, 'train.txt')
    validTXT = os.path.join(args.data_root, 'dataset_splits', args.proto, 'valid.txt')
    args.trainTXT = trainTXT
    args.validTXT = validTXT
    if os.path.isfile(args.trainTXT):
        os.system('rm {}'.format(args.trainTXT))
    if os.path.isfile(args.validTXT):
        os.system('rm {}'.format(args.validTXT))
    
    if test_first:
        print(f'Trying to create label for NTU-RGBD.')
        GeneratLabel(Video_paths[0])

    with Pool(20) as pool:
        for a in tqdm(pool.imap_unordered(GeneratLabel, Video_paths), total=len(Video_paths), desc='Processes'):
            if a is not None:
                pass
    print('Write file list done'.center(80, '*'))

#---------------------------------------------
# video --> Images
#---------------------------------------------
def video_decompre(Video_paths, test_first=False, with_mask=False):
    if test_first:
        print(f'Trying to decompress video {Video_paths[0]}')
        if with_mask:
            video2image_with_mask(Video_paths[0])
            decompre_func = video2image_with_mask
        else:
            video2image(Video_paths[0])
            decompre_func = video2image

        with Pool(20) as pool:
            for a in tqdm(pool.imap_unordered(decompre_func, Video_paths), total=len(Video_paths), desc='Processes'):
                if a is not None:
                    pass
    print('Decompress video done'.center(80, '*'))

#---------------------------------------------
# Images size to (320, 240)
#---------------------------------------------
def image_resize(Video_paths, test_first=False):
    if test_first:
        print(f'Trying to resize video {Video_paths[0]}')
        ResizeImage(Video_paths[0])

    with Pool(40) as pool:
        for a in tqdm(pool.imap_unordered(ResizeImage, Video_paths), total=len(Video_paths), desc='Processes'):
            if a is not None:
                pass
    print('Resize image done'.center(80, '*'))


parser = argparse.ArgumentParser()
parser.add_argument('--proto', default='@CS') # protocol: @CS or @CV
parser.add_argument('--data-root', default='/mnt/workspace/Dataset/NTU-RGBD') # protocol: @CS or @CV

args = parser.parse_args()

Video_paths = glob.glob(os.path.join(data_root, 'nturgb+d_rgb/*.avi'))
assert len(Video_paths) > 0, FileNotFoundError('[error] The file path may be incorrect.')
print('Total videos: {}'.format(len(Video_paths)))

mask_paths = glob.glob(os.path.join(data_root, 'nturgb+d_depth_masked/*'))
assert len(mask_paths) > 0, FileNotFoundError('[error] The file path may be incorrect.')
print('Total Masks: {}'.format(len(mask_paths)))

# video compression
video_decompre(Video_paths, test_first=True, with_mask=True)

# Image resize
image_resize(Video_paths, test_first=True)

# create label file
generate_label(Video_paths, test_first=True)

