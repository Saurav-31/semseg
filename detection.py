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

from bbox_util import *
from data_aug import *
from img_utils import *

dataset_path = '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/'

# data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
data_transform1 = transforms.ToTensor()

aug = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(0.5), RandomTranslate(), RandomRotate(20), RandomShear(), Resize(256)])
# aug = None

dst = PascalDetection(root=dataset_path, is_transform=True, transform = data_transform1, augmentations=aug, img_norm=False, split = 'trainval')
print(len(dst))

classes = dst.pascal_classes()
print(classes)

i = np.random.randint(10000)
img, label = dst[i]

print("Image Shape:", img.shape)
print("No of objects: ", len(label))
h, w = img.shape[1], img.shape[2]

show_bboxes(img.data.numpy().transpose([1, 2, 0]), label, classes)
