# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:19:59 2018

@author: gxjco
"""

import argparse
import os
import scipy.misc
import numpy as np
from model import graph2graph
import tensorflow as tf
import datetime

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', type=int, default=5000, help='number of training epochs')
parser.add_argument('--Ds', type=int, default=200,help='The State Dimention')
parser.add_argument('--Ds_inter', type=int, default=200,help='The State Dimention of inter state')
parser.add_argument('--No', type=int, default=300, help='The Number of Objects')
parser.add_argument('--Nr', type=int, default=89700, help='The Number of Relations')
parser.add_argument('--Dr', type=int, default=1,help='The Relationship Dimension')
parser.add_argument('--Dr_inter', type=int, default=1,help='The Relationship Dimension of inter state')
parser.add_argument('--Dx', type=int, default=3,help='The External Effect Dimension')
parser.add_argument('--De_o', type=int, default=20,help='The Effect Dimension on node')
parser.add_argument('--De_r', type=int, default=20,help='The Effect Dimension on edge')
parser.add_argument('--Mini_batch', type=int, default=1,help='The training mini_batch')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint20',help='models are saved here')
parser.add_argument('--Type', dest='Type', default='test',help='train or test')
args = parser.parse_args()

def main(a = 1):
    print("Processing status: ", a)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    tf.reset_default_graph() 
    with tf.Session() as sess:
        model = graph2graph(sess, Ds=args.Ds,No=args.No,Nr=args.Nr, Dr=args.Dr,Dx=args.Dx,De_o=args.De_o,De_r=args.De_r,Mini_batch=args.Mini_batch,
                       checkpoint_dir=args.checkpoint_dir,epoch=args.epoch,Ds_inter=args.Ds_inter,Dr_inter=args.Dr_inter)
        if args.Type=='train':
           model.train(args)
        if args.Type=='test':
           start = datetime.datetime.now()
           model.test(args)
           end = datetime.datetime.now()
           print (end-start)
if __name__ == '__main__':
    print("Start running")
    #tf.app.run()
    from memory_profiler import memory_usage
    m=memory_usage(main(a = 1), interval=10, timeout=40)
    print(max(m))
    print("Task completed")
