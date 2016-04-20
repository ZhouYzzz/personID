#!/usr/bin/python
'''
Crop cuhk03 images and generate lmdb.
'''
from config import config

CROP_HEIGHT = 240
CROP_WIDTH = 80

import cv2
import numpy as np

def im_read(name, crop=True):
    im = cv2.imread(name)
    assert im is not None
    if crop:
        im = im_crop(im, CROP_HEIGHT, CROP_WIDTH)

    
    im = cv2.resize(im, (120,240))
    im = im.transpose((2,0,1))
    
    '''
    NOTE!!! changed for pythond data layer, need delete
    '''
    #im = np.expand_dims(im,0)
    return im

def im_crop(im, c_h, c_w):
    r_shape = im.shape
    r_h = r_shape[0]
    r_w = r_shape[1]
    if (r_h < c_h):
        im = cv2.resize(im, (100,300))
        r_w = 100; r_h = 300

    if (r_w < c_w):
        im = cv2.resize(im, (100,300))
        r_w = 100; r_h = 300

    off_h = np.random.randint(r_h-c_h+1)
    off_w = np.random.randint(r_w-c_w+1)
    im = im[off_h:off_h+c_h,
            off_w:off_w+c_w,:]
    assert im.shape == (240,80,3)
    return im

