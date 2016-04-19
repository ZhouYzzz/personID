#!/bin/sh

[ $# != 2 ] && { echo MISSING ARGS, should be 2.; exit; }
# check file
[ ! -e $1 ] && { echo MODEL [ $1 ] not exists; exit; }
[ ! -e $2 ] && { echo WEIGHTS [ $2 ] not exists; exit; }

echo TEST TRAINED NET
echo MODEL : [ $1 ]
echo WEIGHTS : [ $2 ]

ROOT=../../

$ROOT/experiments/validation.py.gpu $1 $2
