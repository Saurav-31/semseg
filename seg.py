import matplotlib
matplotlib.use('Agg')
from torchvision import transforms, utils, models
from data_loader_pascal import PascalVOCLoader
from parse_config import *
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import functools
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional
import warnings
import time
import os 
import pickle
# %matplotlib inline
warnings.filterwarnings(action='ignore')

from train import train, val
from loss import dice_loss, accuracy, iou, cross_entropy2d
from fcn import fcn8s
from pspnet import pspnet
from img_utils import showBatchImage, getClassWeights

# conf = parse_cmd_args()
conf = {
  'imsize': 'same' ,
  'seed': 123, 
  'val_split': 0.1,
  'gpu': 2, 
  'num_gpus': 1,
  'max_epochs': 500, 
  'batch_size': 8,
  'img_norm': True,
  'optimizer_config' : {
    'lr': 1.0e-04,
    'weight_decay': 0.0008,
    'momentum': 0.99
  },
  'loss_params': {
    'size_average': True
  },
  'train': False,
  'resume_training': False,
  'pretrained_model': True,
  'dataset_path': '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/',
  'load_model_dir': "./evaluation/model",
  'load_model': "pspnet_101_pascalvoc.pth",
  'exp_dir': 'pspnet_inference',
  'model': 'pspnet' #'fcn8s'
}

os.makedirs("./experiments/", exist_ok=True)

conf['load_model_path'] = os.path.join(conf['load_model_dir'], conf['load_model'])

conf['save_path'] = './experiments/{}'.format(conf['exp_dir'])
conf['model_path'] =  os.path.join(conf['save_path'], "models/")
conf['dump_path'] = os.path.join(conf['save_path'], "img_dmp/")
conf['log_path'] = os.path.join(conf['save_path'], "logs/")

os.makedirs(conf['save_path'], exist_ok=True)
os.makedirs(conf['model_path'], exist_ok=True)
os.makedirs(conf['dump_path'], exist_ok=True)
os.makedirs(conf['log_path'], exist_ok=True)

size = conf['imsize']
# data_transform1 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size),
#                                      transforms.RandomAffine(0, scale=(1.1, 1.4)),
#                                      transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data_transform1 = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
data_transform1 = transforms.ToTensor()
# augs = transforms.RandomAffine(degrees=30, scale=(0.8, 1.4))
augs = None

dst = PascalVOCLoader(root=conf['dataset_path'], is_transform=True, transform = data_transform1, augmentations=augs, img_norm=conf['img_norm'], split = 'trainval')

# img, label  = dst[0]  
# show_sample_img(img, label)

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

print('Train Images: ', len(train_indices))
print('Validation Images: ', len(val_indices))

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SequentialSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dst, batch_size=conf['batch_size'], sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dst, batch_size=conf['batch_size'], sampler=valid_sampler)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.cuda.set_device(6)
torch.cuda.set_device(conf['gpu'])
print("Device: ",  device)
print("GPU: ", conf['gpu'])

dataloaders = {'train': train_loader, 'val': validation_loader}
dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}

if conf['model'] == 'fcn8s':
  model = fcn8s(n_classes=21)
elif conf['model'] == 'pspnet':
  model = pspnet(version="pascal").float()
else:
  print("Model not specified")
  exit()

conf['classWeights'] = torch.Tensor(list(getClassWeights(dst).values()))

# conf['classWeights'][0] = 0 
# print(conf['classWeights'])
# exit()

if conf['resume_training']:
    print("Resuming Training for Model from: ", conf['load_model_path'])
    model.load_state_dict(torch.load(conf['load_model_path']))
    
    pickle_path = '{}/conf.pickle'.format(conf['load_model_dir'])
    with open(pickle_path, "rb") as f:
    	dd = pickle.load(f)
    f.close()
    print(dd)
    conf['best_mean_iou'] = 0.3 #dd['best_mean_iou'] 
    conf['best_epoch'] = dd['best_epoch']

    start_epoch = dd['best_epoch'] + 1
elif conf['pretrained_model']:
    print("Loading Pretrained Model from: ", conf['load_model_path'])
    model.load_state_dict(torch.load(conf['load_model_path']))
    conf['best_mean_iou'] = -10 
    start_epoch = 1
else:
    print("Creating a scratch {} Model with vgg weights".format(conf['model']))
    # vgg16 = models.vgg16(pretrained=True)
    # model.init_vgg16_params(vgg16)
    conf['best_mean_iou'] = -10 
    start_epoch = 1

if conf['num_gpus'] >1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
else:
    model = model.to(device)

img, label  = next(iter(train_loader))
out = model.forward(img.float().to(device))
# print(out.shape)

# showBatchImage(img, label, out, '{}/img_{}.png'.format(conf['dump_path'], 1))

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), **conf['optimizer_config'])
# optimizer = torch.optim.Adam(net.parameters(), **optim_config)

# loss_fn = dice_loss
# loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.BCEWithLogitsLoss(pos_weight = conf['classWeights'].to(device))
loss_fn = nn.CrossEntropyLoss(weight = conf['classWeights'].to(device))
# loss_fn = nn.CrossEntropyLoss(weight = conf['classWeights'].to(device), ignore_index = 0)
# loss_fn = functools.partial(cross_entropy2d, **conf['loss_params'])

if conf['train']:
    epochs = conf['max_epochs']
    print("Starting Training with {} epochs".format(epochs))
else:
    epochs = 1
    print("Starting Evaluation with {} epochs".format(epochs))

num_classes = 21

print("Saving Conf Parameters: {}".format(conf['save_path']))
conf_path = "{}/conf.pickle".format(conf['save_path'])

since = time.time()
for epoch in range(start_epoch, epochs + 1):
    if conf['train']:
        train(model, device, train_loader, optimizer_ft, epoch, conf, loss_fn, num_classes)
    val(model, device, validation_loader, epoch, dataset_sizes['val'], conf, loss_fn, num_classes, dst)
    print("Time Taken for epoch%d: %d sec"%(epoch, time.time()-since))
    since = time.time()
    
    with open(conf_path, "wb") as f:
      pickle.dump(conf, f)
    f.close()

    if epoch % 5 == 0:
        torch.save(model.state_dict(), "{}/FCN8_vgg16_{}.net".format(conf['model_path'], epoch))

