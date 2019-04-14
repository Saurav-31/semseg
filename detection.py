import torch
from PIL import Image
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import collections
from torchvision import transforms, utils, models
from pascalDetectionLoader import PascalDetection
import random 

dataset_path = '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/'
# data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
data_transform1 = transforms.ToTensor()
dst = PascalDetection(root=dataset_path, is_transform=True, transform = data_transform1, augmentations=None, img_norm=False, split = 'trainval')
print(len(dst))

classes = dst.pascal_classes()
print(classes)

i = np.random.randint(10000)
img, label = dst[i]

print("Image Shape:", img.shape)
print("No of objects: ", len(label))
h, w = img.shape[1], img.shape[2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 10))

ax1.imshow(img.data.numpy().transpose([1, 2, 0]))
# Display the image
ax2.imshow(img.data.numpy().transpose([1, 2, 0]))

# Create a Rectangle patch
for obj in label:
    cl, xmin, ymin, wbox, hbox = list(map(float, obj.split()))
    xmin -= wbox/2
    ymin -= hbox/2
    xmin, ymin, wbox, hbox =  xmin*w, ymin*h , wbox*w, hbox*h
    cl = classes[int(cl)+1]
    print(cl)
    rect = patches.Rectangle((xmin,ymin),wbox,hbox,linewidth=2, edgecolor='darkred',facecolor='none')
    ax2.add_patch(rect)
    ax2.text(xmin, ymin, cl,fontsize=15, color ="darkred", fontweight='bold', )

# Add the patch to the Axes
plt.show()
