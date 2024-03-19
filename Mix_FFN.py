from torch import nn
import snntorch as snn
import torch

from config import *

class Mix_FFN(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        
        self.dense1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.lif1 = snn.Leaky(beta= beta, spike_grad=surrogate_grad)
        
        self.conv = nn.Conv2d(channels, channels*expansion, kernel_size=3, groups= channels, padding=1, bias=False)
        self.lif2 = snn.Leaky(beta= beta, spike_grad=surrogate_grad)
        
        #self.relu= nn.ReLU() #might remove this looks useless
        
        self.dense2 = nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False)
        self.lif3 = snn.Leaky(beta= beta, spike_grad=surrogate_grad)
    
    def forward(self,x):
        
        #hidden states initialization
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        x=self.dense1(x)
        spk1, mem1 = self.lif1(x, mem1)
        #print(f"spk1: {spk1.sum()}")
        
        x= self.conv(spk1)
        spk2, mem2 = self.lif2(x, mem2)
        #print(f"spk2: {spk2.sum()}")
        # x= self.relu(spk2)
        
        x=self.dense2(spk2)
        spk3, mem3 = self.lif3(x, mem3)
        #print(f"spk3: {spk3.sum()}")
        
        #print(f"Mix FFN output: {spk3.shape}")
        #print(spk3.sum())
        return spk3 #, mem3

# model= Mix_FFN(8)

# raw_vector = torch.ones(1, 8, 224, 224)*0.5

# rate_coded_vector = torch.bernoulli(raw_vector)

# print(rate_coded_vector.sum())

# out = model(rate_coded_vector)
# print(out.sum())

# #problem here