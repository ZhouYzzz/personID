#!/usr/bin/python
import sys
try:
    sys.argv[1]
except:
    print 'ERROR: No target prototxt specified.'
    exit()

#import numpy as np
sys.path.insert(0, '/home/zhouyz14/caffe/python')
import warnings
warnings.filterwarnings('ignore')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(sys.argv[1], caffe.TEST)
# net.forward()
# im = net.blobs['data'].data[0,:3,:,:].astype(np.uint8)
# label = net.blobs['label'].data[0,0,0,:]
# labela = net.blobs['label_a'].data[0,0,0,:]
# labelb = net.blobs['label_b'].data[0,0,0,:]
# # print net.blobs['feat_b'].data
# print label
# print labela
# print labelb
# import cv2
# im = im.transpose((1,2,0))
# #print im
# cv2.imshow('a',im)
# cv2.waitKey()
