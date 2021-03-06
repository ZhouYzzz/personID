import h5py
import numpy as np
# 1400 classes, 
import sys
sys.path.insert(0,'/home/gpu/zhouyz/caffe/python')
import caffe

import pandas as pd
train = pd.read_csv('pair.txt',sep=' ',header=None)

num_pairs = train.shape[0]

#import cv2
#def read_im(name):
#	im = cv2.imread(name)
#	assert im is not None
#	im = cv2.resize(im, (64,128))
#	im = im.transpose((2,0,1))
#	return im

from process_cuhk03 import im_read


#	f = h5py.File('h5/%04d.h5'%i, 'w')
#	f.create_dataset('data', (3,6,128,64), dtype='uint8')
#	f.create_dataset('label', (3,3,1,1), dtype='uint32')

'''
50400 = 200 * 252
'''

for FI in xrange(252):
    print FI, '/', 252
    F = h5py.File('h5_metric_crop/%04d.h5'%FI, 'w')
    F.create_dataset('data', (200,6,240,80), dtype='uint8')
    F.create_dataset('label', (200,1,1,1), dtype='uint8')
    for i in xrange(200):
        PAIR = train.iloc[i]
        a = im_read('../database/cuhk03/'+PAIR[0])
        b = im_read('../database/cuhk03/'+PAIR[1])

        data = np.vstack((a,b))
        label = np.array([[[PAIR[2]]]])

        F['data'][i] = data
        F['label'][i] = label

    F.close()


# env = lmdb.open('lmdb_metric_crop_small', map_size=int(1e12))
# 
# with env.begin(write=True) as txn:
#     for i in xrange(1000):
#         if (i % 100 == 0):
#             print i,'/',num_pairs
# 
#         PAIR = train.iloc[i]
#         # print PAIR[0], PAIR[1]
#         a = im_read('../database/cuhk03/'+PAIR[0])
#         b = im_read('../database/cuhk03/'+PAIR[1])
#         # print a.shape, b.shape
#         data = np.vstack((a,b))
#         label = PAIR[2]
#         datum = caffe.io.array_to_datum(data, label=label)
#         strid = '{:08}'.format(i)
#         txn.put(strid, datum.SerializeToString())
# 
# env.close()


# for i in xrange(0,1400*36):
# 	cls = i % 1400
# 	sam_sim = train[train[1]==cls].sample(3)
# 	sam_dif = train[train[1]!=cls].sample(1)
#     print sam_sim.iloc[0][0], sam_dif.iloc[0][0], 0
#     print sam_sim.iloc[1][0], sam_sim.iloc[2][0], 1

# for i in xrange(0,1400*36):
#     cls = i%1400
#     sam_sim = train[train[1]==cls].sample(3)
#     sam_dif = train[train[1]!=cls].sample(1)
#     print sam_sim.iloc[0][0], sam_dif.iloc[0][0], 0
#     print sam_sim.iloc[1][0], sam_sim.iloc[2][0], 1
# 
# 
