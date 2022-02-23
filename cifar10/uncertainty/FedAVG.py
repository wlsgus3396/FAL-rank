import copy
import torch
from torch import nn


def FedAvg(w,n):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k]-=w[0][k]
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]*n[i]
            
        w_avg[k] = torch.div(w_avg[k], sum(n))
    return w_avg