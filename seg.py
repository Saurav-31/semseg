import matplotlib
matplotlib.use('Agg')
from torchvision import transforms, utils, models
from data_loader_pascal import PascalVOCLoader
from parse_config import *
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import functools
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional
import warnings
import time
import os 
# %matplotlib inline
warnings.filterwarnings(action='ignore')

from train import train, val
from loss import dice_loss, accuracy, iou, cross_entropy2d
from fcn import fcn8s
from img_utils import showBatchImage

# conf = parse_cmd_args()
conf = {
  'imsize': 224 ,
  'seed': 123, 
  'val_split': 0.1,
  'gpu': 1, 
  'num_gpus': 1,
  'max_epochs': 500, 
  'batch_size': 4,
  'optimizer_config' : {
    'lr': 1.0e-04,
    'weight_decay': 0.0008,
    'momentum': 0.99
  },
  'loss_params': {
    'size_average': True
  },
  'resume_training': False,
  'local_path': '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/'
}

size = conf['imsize']
os.makedirs("./models/", exist_ok=True)
os.makedirs("./img_dmp/", exist_ok=True)
os.makedirs("./logs/", exist_ok=True)

# data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size),
#                                      transforms.RandomAffine(0, scale=(1.1, 1.4)),
#                                      transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data_transform1 = transforms.Compose([transforms.Resize(size), transforms.RandomAffine(0, scale=(0.8, 1.3)),
#                                       transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

#data_transform1 = transforms.Compose([transforms.Resize(256), transforms.RandomAffine(0, scale=(0.8, 1.4)), transforms.ToTensor() ])

dst = PascalVOCLoader(root=conf['local_path'], is_transform=True, transform = data_transform1, augmentations=None, img_norm=True, split = 'trainval')

# img, label  = dst[0]  
# print(img.shape)
# plt.figure(0, figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(img.data.numpy().transpose([1, 2, 0]))

# plt.subplot(1, 2, 2)
# plt.imshow(label)

# plt.show()

# print(img.max(), img.min())
# print(label.max(), label.min())


validation_split = conf['val_split']
shuffle_dataset = True
random_seed = conf['seed']

# Creating data indices for training and validation splits:
dataset_size = len(dst)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

print('Train Images', len(train_indices))
print('Validation Images', len(val_indices))

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dst, batch_size=conf['batch_size'], sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dst, batch_size=conf['batch_size'], sampler=valid_sampler)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.cuda.set_device(6)
torch.cuda.set_device(conf['gpu'])
print("Device: ", device)

dataloaders = {'train': train_loader, 'val': validation_loader}
dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}

model = fcn8s(n_classes=21)

if conf['resume_training']:
	model.load_state_dict(torch.load("./models/FCN8_vgg16_39.net"))
else:
	vgg16 = models.vgg16(pretrained=True)
	model.init_vgg16_params(vgg16)

if conf['num_gpus'] >1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
else:
	model = model.to(device)

img, label  = next(iter(train_loader))
out = model.forward(img.float().to(device))
# print(out.shape)

showBatchImage(img, label, out, './img_dmp/img_{}.png'.format(1))


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), **conf['optimizer_config'])
# optimizer = torch.optim.Adam(net.parameters(), **optim_config)

# loss_fn = dice_loss
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = functools.partial(cross_entropy2d, **conf['loss_params'])

epochs = conf['max_epochs']
print("Starting Training with {} epochs".format(epochs))

num_classes = 21

since = time.time()
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer_ft, epoch, conf, loss_fn, num_classes)
    val(model, device, validation_loader, epoch, dataset_sizes['val'], conf, loss_fn, num_classes)
    # img, label  = next(iter(train_loader))
    # out = model.forward(img.float().to(device))
    # showBatchImage(img, label, out, epoch)
    print("Time Taken for epoch%d: %d sec"%(epoch, time.time()-since))
    since = time.time()
    if epoch % 1 == 0:
        torch.save(model.state_dict(), "./models/FCN8_vgg16_%d.net" % epoch)

