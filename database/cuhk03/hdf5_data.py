import h5py
import numpy as np
# 1400 classes, 

import pandas as pd
train = pd.read_csv('train.txt',sep=' ',header=None)

import cv2
def read_im(name):
	im = cv2.imread(name)
	assert im is not None
	im = cv2.resize(im, (64,128))
	im = im.transpose((2,0,1))
	return im

for i in xrange(1400,2800): # 1400 classes
	if (i % 100 == 0):print 'Processing', i , 'patch'

	cls = i%1400

	f = h5py.File('h5/%04d.h5'%i, 'w')
	f.create_dataset('data', (3,6,128,64), dtype='uint8')
	f.create_dataset('label', (3,3,1,1), dtype='uint32')

	sam_sim = train[train[1]==cls].sample(4)
	sam_dif = train[train[1]!=cls].sample(2)

	# ===== First pair

	im1 = read_im(sam_sim.iloc[0][0])
	lb1 = sam_sim.iloc[0][1]

	im2 = read_im(sam_dif.iloc[0][0])
	lb2 = sam_dif.iloc[0][1]

	data = np.expand_dims(np.vstack((im1,im2)),0)
	label = [[[[lb1]], [[lb2]], [[0]]]]
	label = np.array(label)
	
	f['data'][0] = data
	f['label'][0] = label

	# ===== Second pair

	im1 = read_im(sam_sim.iloc[1][0])
	lb1 = sam_sim.iloc[1][1]

	im2 = read_im(sam_dif.iloc[1][0])
	lb2 = sam_dif.iloc[1][1]

	data = np.expand_dims(np.vstack((im1,im2)),0)
	label = [[[[lb1]], [[lb2]], [[0]]]]
	label = np.array(label)
	
	f['data'][1] = data
	f['label'][1] = label

	# ===== Third pair

	im1 = read_im(sam_sim.iloc[2][0])
	lb1 = sam_sim.iloc[2][1]

	im2 = read_im(sam_sim.iloc[3][0])
	lb2 = sam_sim.iloc[3][1]

	data = np.expand_dims(np.vstack((im1,im2)),0)
	label = [[[[lb1]], [[lb2]], [[1]]]]
	label = np.array(label)
	
	f['data'][2] = data
	f['label'][2] = label
