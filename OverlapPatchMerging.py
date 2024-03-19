import snntorch as snn

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import *

class OverlapPatchMerging(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size, overlap_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size= patch_size, stride= overlap_size, padding=patch_size//2, bias=False),
            snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad),
        )

# model= OverlapPatchMerging(3,8,4,1)

# raw_vector = torch.ones(1, 3, 224, 224)*0.5

# rate_coded_vector = torch.bernoulli(raw_vector)

# print(rate_coded_vector.sum())

# out = model(rate_coded_vector)
# print(out.sum())