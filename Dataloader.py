import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

batch_size = 128
data_path = '/media/iitp/ACER DATA/cityscapes'

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

custom_dataset = ImageFolder(root=data_path, transform=transform)

# Create DataLoader
custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

from snntorch import spikegen

# Iterate through minibatches
data = iter(custom_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=10)

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

spike_data_sample = spike_data[:, 0, 0]

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

anim.save('animation.mp4', writer='ffmpeg')
