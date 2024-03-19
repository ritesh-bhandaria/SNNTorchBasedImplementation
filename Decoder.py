from torch import nn
from typing import List

from DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, out_channels, widths: List[int] , scale_factors: List[int] ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                DecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
        
    def forward(self,features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x=stage(feature)
            new_features.append(x)
    
        return new_features