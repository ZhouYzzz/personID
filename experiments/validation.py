#!/usr/bin/python
import os.path as osp
import numpy as np
BENCHMARK_PATH = osp.join(osp.dirname(__name__),'..','benchmark')
BENCHMARK = ['VIPeR', 'cuhk03']
NET_PATH = osp.join(osp.dirname(__name__),'..','net')
re_identifier_args = {
    'caffe_model':
        NET_PATH+'/metric/deploy.prototxt',
    'caffe_weights':
        NET_PATH+'/metric/_iter_9627.caffemodel'
}
import pandas as pd
def get_validation_set(benchmark='VIPeR', num=0):
    if benchmark not in BENCHMARK:
        raise NameError('No benchmark named [ %s ]'%benchmark)
    if num not in xrange(10):
        raise IndexError('Index [ %d ] out of range (0-9)'%num)
    path = osp.join(BENCHMARK_PATH, benchmark)
    validation = pd.read_csv(
        osp.join(BENCHMARK_PATH,benchmark,'validation%d'%num),
        header=None)
    query = validation[0].values
    gallery = validation[1].values
    return path, query, gallery

def validate_on_benchmark(benchmark='VIPeR'):
    path, query, gallery = get_validation_set(benchmark=benchmark)
    net = caffe.Net(re_identifier_args['caffe_model'],\
                    re_identifier_args['caffe_weights'],\
                    caffe.TEST)
    gallery_feats = list()
    for im in gallery:
        data = get_im_blob(path+'/'+im)
        gallery_feats.append(net.forward(data=data)['ip1'])

    print gallery_feats

def get_im_blob(im):
    import cv2
    im = cv2.imread(im)
    assert im is not None
    im = cv2.resize(im, (128,256))
    im = im.transpose(2,0,1)
    im = np.expand_dims(im, 0)
    return im

def set_up_caffe():
	import sys
	sys.path.insert(0, '/home/zhouyz14/caffe/python')
	import warnings
	warnings.filterwarnings('ignore')
	import caffe
	caffe.set_mode_gpu()
	caffe.set_device(0)
	#log('Caffe set up successfully')
	return caffe

if __name__ == '__main__':
    caffe = set_up_caffe()
    # get_validation_set()
    validate_on_benchmark()
