# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:35:10 2021

@author: Shiyu
"""
import torch
import torch.utils.data
import numpy as np

# Sample code for TwitterNet
############################# TwitterNet
input_edge = np.load('reg_train_new.npy')
target_edge = np.load('mal_train_new.npy')

input_edge = torch.from_numpy(input_edge)
target_edge = torch.from_numpy(target_edge)

input_edge = torch.sigmoid(input_edge)
target_edge = torch.sigmoid(target_edge)

input_edge = input_edge.view(input_edge.shape[0], input_edge.shape[1], input_edge.shape[2], 1)
input_edge = input_edge.repeat(1, 1, 1, 4)
input_edge = input_edge.numpy()

target_edge = target_edge.squeeze()
target_edge = target_edge.numpy()

id_target = np.random.choice(range(target_edge.shape[0]), 500)

target_edge = target_edge[id_target]
input_edge = input_edge[id_target]

np.save('b_emt_input.npy', input_edge)
np.save('b_emt_target.npy', target_edge)
