import collections
from torch.utils.data import Dataset
import os
from scipy import misc 
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
import torch

class PascalVOCLoader(Dataset):
	def __init__(self, root, transform,  split="trainval", is_transform=False, img_size=256, augmentations=None, img_norm=True):
		self.root = os.path.expanduser(root)
		self.split = split
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.img_norm = img_norm
		self.n_classes = 21
		self.mean = np.array([104.00699, 116.66877, 122.67892])
		self.files = collections.defaultdict(list)
		self.img_size = ( img_size if isinstance(img_size, tuple) else (img_size, img_size))

		for split in ["train", "val", "trainval"]:
		    path = os.path.join(self.root, "ImageSets/Segmentation/", split + ".txt")
		    file_list = tuple(open(path, "r"))
		    file_list = [id_.rstrip() for id_ in file_list]
		    self.files[split] = file_list

		self.tf = transform #, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __len__(self):
		return len(self.files[self.split])

	def __getitem__(self, index):
		im_name = self.files[self.split][index]
		im_path = os.path.join(self.root, "JPEGImages", im_name + ".jpg")
		lbl_path = os.path.join(self.root, "SegmentationClass/", im_name + ".png")
		im = Image.open(im_path)
		lbl = Image.open(lbl_path)
		if self.augmentations is not None:
		    im, lbl = self.augmentations(im, lbl)
		if self.is_transform:
		    im, lbl = self.transform(im, lbl)
		return im, lbl

	def transform(self, img, lbl):
		if self.img_size == ('same', 'same'):
		    pass
		else:
		    img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
		    lbl = lbl.resize((self.img_size[0], self.img_size[1]))
		img = self.tf(img)
		lbl = torch.from_numpy(np.array(lbl)).long()
		lbl[lbl == 255] = 0
		return img, lbl

	def get_pascal_labels(self):
		"""Load the mapping that associates pascal classes with label colors

		Returns:
		    np.ndarray with dimensions (21, 3)
		"""
		return np.asarray(
		    [
		        [0, 0, 0],
		        [128, 0, 0],
		        [0, 128, 0],
		        [128, 128, 0],
		        [0, 0, 128],
		        [128, 0, 128],
		        [0, 128, 128],
		        [128, 128, 128],
		        [64, 0, 0],
		        [192, 0, 0],
		        [64, 128, 0],
		        [192, 128, 0],
		        [64, 0, 128],
		        [192, 0, 128],
		        [64, 128, 128],
		        [192, 128, 128],
		        [0, 64, 0],
		        [128, 64, 0],
		        [0, 192, 0],
		        [128, 192, 0],
		        [0, 64, 128],
		    ]
		)

	def encode_segmap(self, mask):
		"""Encode segmentation label images as pascal classes

		Args:
		    mask (np.ndarray): raw segmentation label image of dimension
		      (M, N, 3), in which the Pascal classes are encoded as colours.

		Returns:
		    (np.ndarray): class map with dimensions (M,N), where the value at
		    a given location is the integer denoting the class index.
		"""
		mask = mask.astype(int)
		label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
		for ii, label in enumerate(self.get_pascal_labels()):
		    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
		label_mask = label_mask.astype(int)
		return label_mask

	def decode_segmap(self, label_mask, plot=False):
		"""Decode segmentation class labels into a color image

		Args:
		    label_mask (np.ndarray): an (M,N) array of integer values denoting
		      the class label at each spatial location.
		    plot (bool, optional): whether to show the resulting color image
		      in a figure.

		Returns:
		    (np.ndarray, optional): the resulting decoded color image.
		"""
		label_colours = self.get_pascal_labels()
		r = label_mask.copy()
		g = label_mask.copy()
		b = label_mask.copy()
		for ll in range(0, self.n_classes):
		    r[label_mask == ll] = label_colours[ll, 0]
		    g[label_mask == ll] = label_colours[ll, 1]
		    b[label_mask == ll] = label_colours[ll, 2]
		rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
		rgb[:, :, 0] = r / 255.0
		rgb[:, :, 1] = g / 255.0
		rgb[:, :, 2] = b / 255.0
		if plot:
		    plt.imshow(rgb)
		    plt.show()
		else:
		    return rgb


# import ptsemseg.augmentations as aug
# local_path = '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/'
# bs = 4
# tf = transforms.Compose([transforms.ToTensor()])

# # augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
# dst = PascalVOCLoader(root=local_path, is_transform=True, transform=tf, augmentations=None)
# trainloader = data.DataLoader(dst, batch_size=bs)

# N = int(1000*np.random.random())

# imgs, label = dst[N]
# print(imgs.shape)

# plt.figure(0, figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(imgs.data.numpy().transpose([1, 2, 0]))

# plt.subplot(1, 2, 2)
# plt.imshow(label)

# plt.show()

