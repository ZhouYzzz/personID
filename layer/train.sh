#!/bin/bash
export PYTHONPATH=~/zhouyz/caffe/python:~/zhouyz/personID/layer
~/zhouyz/caffe/build/tools/caffe train -solver=solver.prototxt
