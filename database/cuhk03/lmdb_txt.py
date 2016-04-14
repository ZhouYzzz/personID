import lmdb
import numpy as np
# 1400 classes, 

import pandas as pd
train = pd.read_csv('cls_train.txt',sep=' ',header=None)

import cv2
def read_im(name):
	im = cv2.imread(name)
	assert im is not None
	im = cv2.resize(im, (64,128))
	im = im.transpose((2,0,1))
	return im



# for i in xrange(0,1400*36):
# 	cls = i % 1400
# 	sam_sim = train[train[1]==cls].sample(3)
# 	sam_dif = train[train[1]!=cls].sample(1)
#     print sam_sim.iloc[0][0], sam_dif.iloc[0][0], 0
#     print sam_sim.iloc[1][0], sam_sim.iloc[2][0], 1

for i in xrange(0,1400*36):
    cls = i%1400
    sam_sim = train[train[1]==cls].sample(3)
    sam_dif = train[train[1]!=cls].sample(1)
    print sam_sim.iloc[0][0], sam_dif.iloc[0][0], 0
    print sam_sim.iloc[1][0], sam_sim.iloc[2][0], 1


