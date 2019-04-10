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
from loss import dice_loss, accuracy, iou, cross_entropy2d, calc_metrics
from fcn import fcn8s
from img_utils import showBatchImage, show_sample_img, showBatchImage_decode, getClassWeights

conf = {
  'imsize': 224 ,
  'seed': 123,
  'gpu': 2, 
  'num_gpus': 1, 
  'batch_size': 4,
  'img_norm': False, 
  'loss_params': {
    'size_average': True
  },
  'local_path': '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/',
  'exp_dir': 'exp1',
  'load_model_dir': "./models/FCN8_vgg16_52.net"
 }

os.makedirs("./experiments/", exist_ok=True)

conf['save_path'] = './experiments/{}'.format(conf['exp_dir'])
conf['dump_path'] = os.path.join(conf['save_path'], "val_imgs/")

os.makedirs(conf['save_path'], exist_ok=True)
os.makedirs(conf['dump_path'], exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(conf['gpu'])

data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
dst = PascalVOCLoader(root=conf['local_path'], is_transform=True, transform = data_transform1, augmentations=None, img_norm=conf['img_norm'], split = 'val')
validation_loader = torch.utils.data.DataLoader(dst, batch_size=conf['batch_size'])

conf['classWeights'] = getClassWeights(dst)

# img, label  = dst[0]  
# show_sample_img(img, label)

model = fcn8s(n_classes=21)
model = model.to(device)
model.load_state_dict(torch.load(conf['load_model_dir']))

model.eval()

# img, label  = next(iter(validation_loader))
# out = model.forward(img.float().to(device))
# out_pred = out.data.max(1)[1].to(device)

# i = 0
# print("Pixel Accuracy: ", ev.pixel_accuracy(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))
# print("Mean Accuracy: ", ev.mean_accuracy(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))
# print("Mean IOU: ", ev.mean_IU(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))
# print("Frequency Weighted IOU: ", ev.frequency_weighted_IU(label[i].cpu().data.numpy(), out_pred[i].cpu().data.numpy()))

# showBatchImage(img, label, out, './val_imgs/img_{}.png'.format(1))
# showBatchImage_decode(img, label, out, dst, './val_imgs/img_{}.png'.format("sample2"))

pix_acc = 0
mean_acc = 0
mean_iou = 0
freq_w_iou = 0

from tqdm import tqdm 
nbatches = len(validation_loader)
pbar = tqdm(total = nbatches)

with torch.no_grad():
    for batch_idx, sample in enumerate(validation_loader):
        data, target = sample[0].float().to(device), sample[1].float().to(device)
        output = model(data)
        out_pred = output.data.max(1)[1].to(device)

        pix_acc_batch, mean_acc_batch, mean_iou_batch, freq_w_iou_batch = calc_metrics(target, out_pred)
        pix_acc += pix_acc_batch
        mean_acc += mean_acc_batch
        mean_iou += mean_iou_batch
        freq_w_iou += freq_w_iou_batch

        showBatchImage_decode(data, target, output, dst, '{}/img_{}.png'.format(conf['dump_path'], batch_idx))
        pbar.update(1)

print("Pixel Accuracy: ", pix_acc/nbatches)
print("Mean Accuracy: ", mean_acc/nbatches)
print("Mean IOU: ", mean_iou/nbatches)
print("Frequency Weighted IOU: ", freq_w_iou/nbatches)


