from torch import nn
from EfficientMultiHeadedAttention import EfficientMultiHeadedAttention
from Mix_FFN import Mix_FFN
from torchvision.ops import StochasticDepth

class ResidualAdd(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, **kwargs):
    out = self.fn(x, **kwargs)
    x=x+out
    return x

class SegFormerEncoderBlock(nn.Sequential):
  def __init__(
      self,
      channels: int,
      reduction_ratio: int = 1,
      num_heads: int = 8,
      mlp_expansion: int = 4,
      drop_path_prob: float = 0.0,
  ):

    super().__init__(
        ResidualAdd(
            nn.Sequential(
                EfficientMultiHeadedAttention(channels, reduction_ratio, num_heads)
            )
        ),
        ResidualAdd(
            nn.Sequential(
                Mix_FFN(channels, expansion = mlp_expansion),
                #StochasticDepth(p=drop_path_prob, mode="batch")
            )
        ),
    )