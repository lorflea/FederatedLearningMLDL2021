#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import matplotlib.pyplot as plt


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

  
def FedAvg2(w, l):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[0][k], l[0]/sum(l))

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += torch.mul(w[i][k], l[i]/sum(l))
        #w_avg[k] = torch.div(w_avg[k], len(w))
    
    return w_avg