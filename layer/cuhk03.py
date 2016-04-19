import os
import pandas as pd
import numpy as np
import cv2
from time import time
from multiprocessing import Pool
from random import getrandbits

def imread(name):
    im = cv2.imread(name)
    assert im is not None
    return im.transpose(2,0,1)

class MtReader():
    def __init__(self, target, labels):
        self.target = target
        self.labels = labels

    def __call__(self, idx):
        self.target[idx,:,:,:] = imread(labels[idx])
        return None

class CUHK03():
    def __init__(self, list):
        df = pd.read_csv(list, sep=' ', header=None)
        self.fnames = df[0].values
        self.labels = df[1].values
        self.num = df.shape[0]
        #self.num = 10000
        im = cv2.imread(self.fnames[0])
        assert im is not None
        (self.H ,self.W, self.C) = im.shape
        self.LOG('Shape'+str(im.shape)+'Num'+str(self.num))
        self.data = None
        self.clsidx = self.idx2cls()
        self.numcls = 1467
        # when generate pairs, the bound for choice
        self.bound = [0,200]

    def idx2cls(self):
        clsidx = list()
        for i in xrange(1467):
            clsidx.append(np.where(self.labels == i)[0].tolist())

        return clsidx

    def cls2idx(self, cls):
        return self.clsidx[cls]

    def choice(self, cls, num):
        return np.random.choice(self.cls2idx(cls), num)

    def crop(self, idx, W=120, H=240):
        #t = time()
        off_h = np.random.randint(self.H-H+1)
        off_w = np.random.randint(self.W-W+1)
        im = self.data[idx,:,off_h:off_h+H,off_w:off_w+W].copy()
        #print '[crop]',im.shape
        #print time()-t
        return im

    def get(self, idx):
        return self.data[idx,:,:,:]

    @classmethod
    def mtread(self, idx):
        self.data[idx,:,:,:] = imread(self.fnames[idx])
        return None

    def mtload(self):
        self.data = np.zeros([self.num,self.C,self.H,self.W],np.uint8)
        pool = Pool(32)
        mtreader = MtReader(self.data, self.labels)
        pool.map(mtreader ,xrange(self.num))

    def load(self):
        self.LOG('Loading imgs to memory')
        if self.data is not None:
            return False
        self.data = np.zeros([self.num,self.C,self.H,self.W],np.uint8)
        for i in xrange(self.num):
            self.data[i,:,:,:] = imread(self.fnames[i])

        self.LOG('Imgs read successfully')

    def gensim(self, cls):
        '''sim'''
        [i1, i2] = self.choice(cls, 2)
        #print i1, i2
        return np.vstack((self.crop(i1), self.crop(i2)))

    def gendif(self, cls):
        '''dif'''
        [i1] = self.choice(cls, 1)
        [i2] = self.choice(self.rand_not_cls(cls), 1)
        #print i1,i2
        return np.vstack((self.crop(i1), self.crop(i2)))

    def genpair(self, cls):
        '''genpair'''
        if bool(getrandbits(1)):
            #print '[S]'
            return (1,self.gensim(cls))
        else:
            #print '[D]'
            return (0,self.gendif(cls))

    def rand_not_cls(self, cls):
        '''not the cls'''
        r = np.random.randint(*self.bound)
        if (r==cls):
            return self.rand_not_cls(cls)
        else:
            return r

    def LOG(self, msg):
        print b'\033[92m'+'[CUHK03]:'+'\033[0m', msg

#cuhk03 = CUHK03('labeled.txt')
#cuhk03.load()
#print cuhk03.genpair(2)
#print cuhk03.gensim(2)
#print cuhk03.choice(3,5)

