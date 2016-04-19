import sys
sys.path.append('/home/gpu/zhouyz/personID/layer')
import caffe
import yaml
DEBUG = True
import numpy as np
import cv2
from process_cuhk03 import im_read, im_crop
import pandas as pd
from time import time
from threading import Thread
from multiprocessing import Pool
from cuhk03 import CUHK03

BATCHSIZE = 100

class ImBuffer():
    def __init__(self, imlist):
        self.buffer = np.zeros([num_im, 3, 256, 128], dtype=np.uint8)
        self.imlist = imlist

    def load(self, crop=False):
        self.clear()
        LOG('loading buffer')
        t = time()
        for i in xrange(num_im):
            self.buffer[i] = cv2.imread(imlist[i]).transpose((2,0,1))

        if crop: self.prepare_for_crop()

        # self._transpose()
        print 'Load took', time() - t

    def _transpose(self):
        for key in self.buffer.keys():
            self.buffer[key].transpose((2,0,1))

    def prepare_for_crop(self, H=240, W=120):
        LOG('Pre for Crop')
        for im in imlist:
            t = self.buffer[im]

            r_shape = t.shape
            r_h = r_shape[0]
            r_w = r_shape[1]
            if (r_h < H):
                t = cv2.resize(t, (W+20,H+60))
                r_w = W; r_h = H

            if (r_w < W):
                t = cv2.resize(t, (W+20,H+60))
                r_w = W; r_h = H

            self.buffer[im] = t

    def get(self, im):
        assert im in self.imlist
        return self.buffer[im]

    def clear(self):
        del self.buffer
        self.buffer = dict()

def image_processor(im):
    return im_crop(im, 240, 80, LOG=False, TIME=False)

def generate_pairs():
    return (im1, im2)

def advance_batch(result, buffer, pool):
    t = time()
    data = np.zeros([BATCHSIZE,6,240,120])
    label = np.zeros([BATCHSIZE,1,1,1])
    cls = np.random.randint(200, size=BATCHSIZE)
    for i in xrange(BATCHSIZE):
        pair = buffer.genpair(i)
        data[i] = pair[1]
        label[i,:,:,:] = pair[0]

    result['data'] = data
    result['label'] = label
    LOG('process take %.4f s.'%(time()-t))

class BatchAdvancer():
    def __init__(self, result, buffer, pool):
        self.result = result
        self.buffer = buffer
        self.pool = pool
        # print self.result
        pass

    def __call__(self):
        #print self.result
        return advance_batch(self.result, self.buffer, self.pool)

class MetricDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        if DEBUG: LOG('setup')
        #print yaml.load(self.param_str)
        assert len(bottom) == 0
        assert len(top) == 2
        f = self.param_str
        self.l = pd.read_csv(f,header=None)
        self.path = '/home/gpu/zhouyz/cuhk-03/'

        self.db = CUHK03('labeled.txt')
        self.db.load()

        #ImBuffer.load(imlist)
        # THREAD
        self.thread_result = {}
        self.thread = None
        #pool_size = 16
        self.N = BATCHSIZE # batchsize
        #self.pool = Pool(processes=pool_size)
        self.pool = None
        # ADVANCER
        self.batch_advancer = \
            BatchAdvancer(self.thread_result, self.db, self.pool)

        self.dispatch_worker()
        self.join_worker()

        pass

    def reshape(self, bottom, top):
        #if DEBUG: LOG('reshape')
        top[0].reshape(*(self.N,6,240,120))
        top[1].reshape(*(self.N,1,1,1))
        pass

    def forward(self, bottom, top):
        #if DEBUG: LOG('forward')
        if self.thread is not None:
            self.join_worker()

        '''do forward'''
        top[0].data[...] = self.thread_result['data']
        top[1].data[...] = self.thread_result['label']

        self.dispatch_worker()
        pass

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        pass



def LOG(msg):
    print bcolors.OKGREEN+'[MetricDataLayer]:'+bcolors.ENDC, msg

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    #i = im_read(self.path+self.l.iloc[0][0])
        #top[0].data[0,...] = i# np.ones([2,2,2])
        #top[1].data[0,0,0,0] = 0
        #LOG(self.thread_result.keys())
        #print self.thread_result['data'].shape
        #print self.thread_result['data'][0].shape
