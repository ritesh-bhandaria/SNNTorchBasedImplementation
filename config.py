from snntorch import surrogate

#SNN LIF neuron paramters
beta=0.125
surrogate_grad = surrogate.fast_sigmoid(slope=25)


#seg former paramerters
in_channels=3
widths=[64, 128, 256, 512]
depths=[3, 4, 6, 3]
all_num_heads=[1, 2, 4, 8]
patch_sizes=[7, 3, 3, 3]
overlap_sizes=[4, 2, 2, 2]
reduction_ratios=[8, 4, 2, 1]
mlp_expansions=[4, 4, 4, 4]
decoder_channels=256
scale_factors=[8, 4, 2, 1]
num_classes=100

#model hyperparameters
num_steps=50
batch_size=32
learning_rate= 0.001
lr_factor = 0.1
lr_patience = 3
early_stopping_patience = 10
max_epochs= 1

#model weight saving path
path= "SpikingSegFormer.pth"