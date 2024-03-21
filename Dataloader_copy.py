from torchvision.datasets import Cityscapes
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from config import num_steps
import torch

from snntorch import spikegen

#these classes are going to be segmentated total 20 classes
ignore_index = 255
void_classes= [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,-1]
valid_classes= [ignore_index, 7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
class_names = ['unlabeled',
               'road',
               'sidewalk',
               'building',
               'wall',
               'fence',
               'pole',
               'traffic light',
               'traffic sign',
               'vegetation',
               'terrain',
               'sky',
               'person',
               'rider',
               'car',
               'truck',
               'bus',
               'train',
               'motorcycle',
               'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)

#in later stage we might want to get the segements as different color coded so this is for it
colors =[
    [  0,   0,   0],
    [128,  64, 128],
    [244,  35, 232],
    [ 70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [ 70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [  0,   0, 142],
    [  0,   0,  70],
    [  0,  60, 100],
    [  0,  80, 100],
    [  0,   0, 230],
    [119,  11,  32],
]

label_colors = dict(zip(range(n_classes), colors))

def encode_segmap(mask):
    '''
    online mila tha 
    remove unwanted classes and rectify the labels of wanted classes
    '''
    for void_c in void_classes:
        mask[mask == void_c] = ignore_index
    for valid_c in valid_classes:
        mask[mask == valid_c] = class_map[valid_c]
    
    return mask

def decode_segmap(temp):
    '''
    ye bhi online mila tha
    convert greyscale to color
    ye to use nahi kara hai 
    '''
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0] 
        g[temp == l] = label_colors[l][1] 
        b[temp == l] = label_colors[l][2]
    
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = r/255.0 
    rgb[:,:,1] = g/255.0 
    rgb[:,:,2] = b/255.0 
    
    return rgb



class AdjustGamma:
    '''
    image bohot dark aa rahi thi to ye laga diya hai
    thoda washed ho gayi hai image par iski wajah se
    '''
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain
    
    def __call__(self, image, mask):
        img = np.transpose(image,(2,0,1))
        gamma_tensor = torchvision.transforms.functional.adjust_gamma(torch.from_numpy(img), self.gamma, self.gain)
        img = np.transpose(gamma_tensor.numpy(), (1,2,0))
        return {'image': img, 'mask': mask}

class SpikeEncoding:
    '''
    rate coding ka code
    '''
    def __call__(self, image, mask):
        image = image.float()  # Convert to float
        mask = mask.float()    # Convert to float
        out_img = spikegen.rate(image, num_steps=num_steps)
        out_mask = spikegen.rate(mask, num_steps=num_steps)
        # out_img = out_img.bool()      #IMPORTNAT BUT NOT IMPLEMENTED
        # out_mask = out_mask.bool()
        return {'image': out_img, 'mask': out_mask}
    
class normalizeSeg:
    '''
    segmap me pata nahi normalized values nahi aa rahi thi
    to ye ek alag se bhi bana diya 
    '''
    def __call__(self, image, mask):
        normalized_seg = (mask-torch.min(mask))/(torch.max(mask)-torch.min(mask))
        return {'image': image, 'mask': normalized_seg}

class encode:
    '''
    upar wala encode segmap function use kara hai mask par
    '''
    def __call__(self, image, mask):
        final = encode_segmap(mask)
        return {'image': image, 'mask': final}
'''
yaha par transforms hai 
'''
transform = A.Compose(
    [
        A.Resize(224,224),
        AdjustGamma(gamma=0.63),
        A.Normalize(mean = (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225), max_pixel_value = float(225)),
        ToTensorV2(),
        encode(),
        normalizeSeg(),
        SpikeEncoding(),
    ]
)


from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

class data_transform(Cityscapes):
    '''
    isko samajhne ki koi zarurat nahi 
    waise bhi source code uthaya hai bas
    par ye wala dono (image, segmap ) ko return karta hai ek saath
    '''
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        
        targets: Any = []
        for i,t in enumerate(self.target_type):
            if t == 'polygon':
                target = self.load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]
        

        if self.transforms is not None :
            transformed=transform(image=np.array(image), mask=np.array(target))
            return transformed['image'], transformed['mask']
        return image, target

'''
aise data aa ajyega

dataset = data_transform('/media/iitp/ACER DATA1/cityscapes', split='val', mode='fine', target_type='semantic', transforms=transform)
img, seg = dataset[20]
print(img.shape, seg.shape)

print(img.sum())
print(seg.sum())
print(img.dtype)
print(seg.dtype)

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

ch=['r','g','b']
for i in range (0,3):
    img_sample = img[:, i]
    #print(img.size())
    fig, ax = plt.subplots()
    anim = splt.animator(img_sample, fig, ax)
    HTML(anim.to_html5_video())
    anim.save(f"spike_mnist_{ch[i]}.gif")


mask_sample = seg[:]
fig, ax = plt.subplots()
anim = splt.animator(mask_sample, fig, ax)
HTML(anim.to_html5_video())
anim.save(f"mask_sample.gif")
'''

from torch.utils.data import DataLoader
import pytorch_lightning as pl

class CustomCollate:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, batch):
       # Assuming batch is a list of tensors [image, mask]
       images = [item[0] for item in batch]
       masks = [item[1] for item in batch]
       
       # Stack images and masks along dimension 1
       images_stacked = torch.stack(images, dim=1)
       masks_stacked = torch.stack(masks, dim=1)
       
       return [images_stacked, masks_stacked]


class GetData(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage= None):
        self.train_dataset = data_transform(root='../../cityscapes', split='train', mode='fine', target_type='semantic', transforms=transform)
        self.val_dataset = data_transform(root='../../cityscapes', split='val', mode='fine', target_type='semantic', transforms=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    

