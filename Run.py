from SegFormer import SegFormer
import torch
from config import *




test_segformer = SegFormer(
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
)

num_steps = 1

raw_vector = torch.ones(num_steps, 1, 3, 224, 224)*1

rate_coded_vector = torch.bernoulli(raw_vector)

print(rate_coded_vector.sum())
# result = test_segformer(rate_coded_vector)

# loop for num steps
mem_rec = []
spk_rec = []

for step in range(num_steps):
    spk_out= test_segformer(rate_coded_vector[step])
    print(spk_out.sum())
    print(f"output shape: {spk_out.shape}") #everything is 0 BIG ISSUE
    spk_rec.append(spk_out)
    # mem_rec.append(mem_out)

#only the spike train is the output not mem output you can have it uncomment it from retrun statement
# can't get summary