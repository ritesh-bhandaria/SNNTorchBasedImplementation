from SegFormerEncoder import SegFormerEncoder
from Decoder import Decoder
from SegmentationHead import SegmentationHead
from torch import nn

class SegFormer(nn.Module):
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
        decoder_channels,
        scale_factors,
        num_classes,
        drop_prob : float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = Decoder(decoder_channels, widths[::-1], scale_factors)
        self.seghead = SegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.seghead(features)
        return segmentation