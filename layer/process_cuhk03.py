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
    if im is None:
        raise NameError('No file [%s]'%name)
    if crop:
        im = im_crop(im, CROP_HEIGHT, CROP_WIDTH)

    im = im.transpose((2,0,1))
    #im = np.expand_dims(im,0)
    return im

from time import time

def im_crop(im, c_h, c_w, LOG=False, TIME=False):
    r_shape = im.shape
    r_h = r_shape[0]
    r_w = r_shape[1]
    t1 = time()
    if (r_h < c_h):
        #print 'Height'
        if LOG: print 'H <'
        im = cv2.resize(im, (100,300))
        #assert im.shape[0]==c_h
        r_w = 100; r_h = 300

    if (r_w < c_w):
        #print 'Width'
        if LOG: print 'W <'
        im = cv2.resize(im, (100,300))
        #assert im.shape[1]==c_w
        r_w = 100; r_h = 300

    t2 = time()
    if TIME: print 'resize', t2 -t1
    #print im.shape
    off_h = np.random.randint(r_h-c_h+1)
    off_w = np.random.randint(r_w-c_w+1)

    t3 = time()
    if TIME: print 'random', t3-t2
    # print off_h, off_w
    im = im[off_h:off_h+c_h,
            off_w:off_w+c_w,:]

    t4 = time()
    if TIME: print 'crop', t4-t3
    assert im.shape == (240,80,3)
    return im

