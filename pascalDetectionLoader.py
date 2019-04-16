from PIL import Image
import os
import numpy as np 
from torch.utils.data import Dataset
import collections
from img_utils import *
import matplotlib.pyplot as plt 

class PascalDetection(Dataset):
    def __init__(self, root, transform,  split="trainval", is_transform=False, img_size='same', augmentations=None, img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.aug = augmentations
        self.img_norm = img_norm
        self.n_classes = 21
        self.classes = self.pascal_classes()
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = ( img_size if isinstance(img_size, tuple) else (img_size, img_size))		
        for split in ["train", "val", "trainval"]:
            path = os.path.join(self.root, "ImageSets/Main/", split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        self.tf = transform #, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        img_path = os.path.join(self.root, "JPEGImages", im_name + ".jpg")
        label_path = os.path.join(self.root, "labels_2012/", im_name + ".txt")

        im = np.array(Image.open(img_path))
        h, w, channels = im.shape

        with open(label_path, "r") as f:
            lbl = f.readlines()
        
        lbl = np.array([list(map(float, x.split(" "))) for x in np.array(lbl)])
        # print(clss.shape, bbox.shape)

        if self.aug is not None:
            bbox = lbl[:, 1:]
            clss = lbl[:, 0]
            bbox = convert_center_to_coord(im, bbox)
            # print("After Conversion", bbox)

            im1, lbl1 = self.aug(im.copy(), bbox.copy())
            # print("After Augmentation", lbl1)

            if lbl1.shape[0] == clss.shape[0]:
                lbl = merge_cl_bbox(clss, lbl1)
                im = im1
            else:
                lbl = merge_cl_bbox(clss, bbox)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl

    def transform(self, img, lbl):
        h, w, channels = np.array(img).shape
        
        if self.img_size == ('same', 'same'):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = self.tf(img)
        
#         xratio = w/self.img_size[0]
#         yratio = h/self.img_size[1]
        
        return img, lbl

    def augmentations(self, img, lbl):
        pass

    def pascal_classes(self):
        classes = ['background', 'aeroplane', 'bicycle', 'bird',
                    'boat','bottle','bus','car','cat','chair','cow','diningtable',
                    'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

        return classes
