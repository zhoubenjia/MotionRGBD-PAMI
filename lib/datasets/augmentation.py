'''
This file is modified from:
https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/affine.py
'''
import PIL
from PIL import ImageFilter, ImageOps, Image
import os, glob
import math, random
import numpy as np
import logging
import cv2
from scipy.ndimage.filters import gaussian_filter
import numbers

class Sequential(object):
    """
    Composes several augmentations together.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.rand = random_order

    def __call__(self, clip):
        if self.rand:
            rand_transforms = self.transforms[:]
            random.shuffle(rand_transforms)
            for t in rand_transforms:
                clip = t(clip)
        else:
            for t in self.transforms:
                clip = t(clip)

        return clip

class Sometimes(object):
    """
    Applies an augmentation with a given probability.
    Args:
        p (float): The probability to apply the augmentation.
        transform (an "Augmentor" object): The augmentation to apply.
    Example: Use this this transform as follows:
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        sometimes(va.HorizontalFlip)
    """

    def __init__(self, p, transform):
        self.transform = transform
        if (p > 1.0) | (p < 0.0):
            raise TypeError('Expected p to be in [0.0 <= 1.0], ' +
                            'but got p = {0}'.format(p))
        else:
            self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            clip = self.transform(clip)
        return clip

class RandomTranslate(object):
    """
      Shifting video in X and Y coordinates.
        Args:
            x (int) : Translate in x direction, selected
            randomly from [-x, +x] pixels.
            y (int) : Translate in y direction, selected
            randomly from [-y, +y] pixels.
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.x_move = random.randint(-self.x, +self.x)
        self.y_move = random.randint(-self.y, +self.y)

    def __call__(self, clip):
        x_move = self.x_move
        y_move = self.y_move

        if isinstance(clip, np.ndarray):
            rows, cols, ch = clip.shape
            transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
            return cv2.warpAffine(clip, transform_mat, (cols, rows))
        elif isinstance(clip, Image.Image):
            return clip.transform(clip.size, Image.AFFINE, (1, 0, x_move, 0, 1, y_move))
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip)))

class RandomResize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate

        self.interpolation = interp
        self.scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

    def __call__(self, clip):
        if isinstance(clip, np.ndarray):
            im_h, im_w, im_c = clip.shape
        elif isinstance(clip, PIL.Image.Image):
            im_w, im_h = clip.size
        else:
            raise TypeError(f'Unknow image type {type(clip)}')
        new_w = int(im_w * self.scaling_factor)
        new_h = int(im_h * self.scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip, np.ndarray):
            return scipy.misc.imresize(clip, size=(new_h, new_w),interp=self.interpolation)
        elif isinstance(clip, PIL.Image.Image):
            return clip.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation))
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip)))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC

class RandomShear(object):
    """
    Shearing video in X and Y directions.
    Args:
        x (int) : Shear in x direction, selected randomly from
        [-x, +x].
        y (int) : Shear in y direction, selected randomly from
        [-y, +y].
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_shear = random.uniform(-self.x, self.x)
        self.y_shear = random.uniform(-self.y, self.y)

    def __call__(self, clip):
        x_shear, y_shear = self.x_shear, self.y_shear
        if isinstance(clip, np.ndarray):
            rows, cols, ch = clip.shape
            transform_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
            return cv2.warpAffine(clip, transform_mat, (cols, rows))
        elif isinstance(clip, PIL.Image.Image):
            return clip.transform(img.size, PIL.Image.AFFINE, (1, x_shear, 0, y_shear, 1, 0))
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip)))
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.
    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """
    def __init__(self, ratio=100):
        self.ratio = ratio
        self.flag = True
        self.noise = None

    def __call__(self, clip):
        is_PIL = isinstance(clip, PIL.Image.Image)
        if is_PIL:
            clip = np.asarray(clip)

        # if self.flag:
        #     img = clip.astype(np.float)
        #     img_shape = img.shape
        #     self.noise = np.random.randint(self.ratio, size=img_shape)
        #     img = np.where(self.noise == 0, 255, img)
        #     clip = img.astype(np.uint8)
        #     self.flag = False
        img = clip.astype(np.float)
        img_shape = img.shape
        self.noise = np.random.randint(self.ratio, size=img_shape)
        img = np.where(self.noise == 0, 255, img)
        clip = img.astype(np.uint8)

        if is_PIL:
            return PIL.Image.fromarray(clip)
        else:
            return clip

class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size
        self.flag = True
        self.w1, self.h1 = None, None
        self.crop_w, self.crop_h = None, None

    def __call__(self, clip):
        if self.flag:
            crop_w, crop_h = self.size
            self.crop_w, self.crop_h = crop_w, crop_h
            if isinstance(clip, np.ndarray):
                im_h, im_w, im_c = clip.shape
            elif isinstance(clip, PIL.Image.Image):
                im_w, im_h = clip.size
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip)))
            if crop_w > im_w:
                crop_w = im_w
            if crop_h > im_h:
                crop_h = im_h
            # if crop_w > im_w or crop_h > im_h:
                # error_msg = ('Initial image size should be larger then' +
                #             'cropped size but got cropped sizes : ' +
                #             '({w}, {h}) while initial image is ({im_w}, ' +
                #             '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                #                             h=crop_h))
                # raise ValueError(error_msg)

            self.w1 = random.randint(0, im_w - crop_w)
            self.h1 = random.randint(0, im_h - crop_h)
            self.flag = False
            
        w1, h1 = self.w1, self.h1
        crop_w, crop_h = self.crop_w, self.crop_h
        if isinstance(clip, np.ndarray):
            return clip[h1:h1 + crop_h, w1:w1 + crop_w, :]
        elif isinstance(clip, PIL.Image.Image):
            return clip.crop((w1, h1, w1 + crop_w, h1 + crop_h))
