from torch import nn
import torch
from typing import Iterable
from SegFormerEncoderStage import SegFormerEncoderStage
import numpy as np

def chunks(data: Iterable, sizes):
  curr=0
  for size in sizes:
    chunk = data[curr: curr+size]
    curr+=size
    yield chunk

class SegFormerEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      widths,
      depths,
      all_num_heads,
      patch_sizes,
      overlap_sizes,
      reduction_ratios,
      mlp_expansions,
      drop_prob = 0.0,
  ):
    super().__init__()

    drop_probs = [x.item() for x in torch.linspace(0,drop_prob,sum(depths))]
    self.stages = nn.ModuleList(
        [
            SegFormerEncoderStage(*args)
            for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
        ]
    )

  def forward(self,x):
    features = []
    for stage in self.stages:
      x=stage(x)
      features.append(x)
      
    #print(f"encoder output: {features.shape}")
    return features