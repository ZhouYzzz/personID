#!/usr/bin/python
import numpy as np

for i in xrange(10):
    select = np.random.choice(np.arange(1436),316,False)
    f = open('validation%d'%i, 'w')
    for x in select:
        f.write('img/%04d_01.jpg,img/%04d_06.jpg\n'%(x,x))


