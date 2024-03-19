from torch import nn
import snntorch as snn
import torch

from config import *

class SegmentationHead(nn.Module):
    def __init__(self, channels, num_classes, num_features = 4):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.num_features = num_features
    
        self.dense1 = nn.Conv2d(channels*num_features, channels, kernel_size=1, bias=False)
        self.lif1 = snn.Leaky(beta= beta, spike_grad=surrogate_grad)
    
        #self.relu = nn.ReLU()
        #self.bn = nn.BatchNorm2d(channels)
    
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.lif2 = snn.Leaky(beta= beta, spike_grad=surrogate_grad)

    def forward(self,x):
      
        #hidden state initialization
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
    
        x=torch.cat(x, dim=1)
    
        x=self.dense1(x)
        spk1, mem1 = self.lif1(x, mem1)
    
        #x=self.relu(x)
        #x=self.bn(x)
    
        x=self.predict(spk1)
        spk2, mem2 = self.lif2(x, mem2)

        return spk2 #, mem2
