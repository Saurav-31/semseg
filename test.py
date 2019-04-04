import torch
import matplotlib
matplotlib.use('Agg')
from torchvision import transforms, utils, models
from data_loader_pascal import PascalVOCLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
import time
import os 
# %matplotlib inline
warnings.filterwarnings(action='ignore')

from train import train, val
from loss import dice_loss, accuracy, iou, cross_entropy2d
from fcn import fcn8s
from img_utils import showBatchImage, show_sample_img, showBatchImage_decode
import eval_metrics as ev

conf = {
  'imsize': 224 ,
  'seed': 123,
  'gpu': 2, 
  'num_gpus': 1, 
  'batch_size': 4,
  'loss_params': {
    'size_average': True
  },
  'local_path': '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/'
 }

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(conf['gpu'])

data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
dst = PascalVOCLoader(root=conf['local_path'], is_transform=True, transform = data_transform1, augmentations=None, img_norm=True, split = 'val')
validation_loader = torch.utils.data.DataLoader(dst, batch_size=conf['batch_size'])

# img, label  = dst[0]  
# show_sample_img(img, label)

model = fcn8s(n_classes=21)
model = model.to(device)
model.load_state_dict(torch.load("./models/FCN8_vgg16_39.net"))

model.eval()
os.makedirs("./val_imgs/", exist_ok=True)

img, label  = next(iter(validation_loader))
out = model.forward(img.float().to(device))
out_pred = out.data.max(1)[1].to(device)

i = 0
print("Pixel Accuracy: ", ev.pixel_accuracy(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))
print("Mean Accuracy: ", ev.mean_accuracy(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))
print("Mean IOU: ", ev.mean_IU(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))
print("Frequency Weighted IOU: ", ev.frequency_weighted_IU(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))

# print(out.shape)

# showBatchImage(img, label, out, './val_imgs/img_{}.png'.format(1))
showBatchImage_decode(img, label, out, dst, './val_imgs/img_{}.png'.format("sample"))

# from tqdm import tqdm 
# nbatches = len(validation_loader)
# pbar = tqdm(total = nbatches)

# with torch.no_grad():
#     for batch_idx, sample in enumerate(validation_loader):
#         data, target = sample[0].float().to(device), sample[1].float().to(device)
#         output = model(data)
#         showBatchImage_decode(data, target, output, dst, './val_imgs/img_{}.png'.format(batch_idx))
#         pbar.update(1)


