# import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split

batch_size = 128
data_path = '/media/iitp/ACER DATA1/cityscapes'

# Define a transform
transform = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

custom_dataset = ImageFolder(root=data_path, transform=transform)

# Create DataLoader
dataset_size = len(custom_dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
test_size = dataset_size - train_size  # Remaining for testing

# Split dataset into training and testing sets
train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

# Create DataLoaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

from snntorch import spikegen

# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=55)

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

spike_data_sample = spike_data[:, 0, 0]

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

anim.save('./animation.mp4', writer='ffmpeg')
