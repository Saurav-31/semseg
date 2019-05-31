import matplotlib
from torchvision import transforms, utils, models
from data_loader_pascal import PascalVOCLoader
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings
warnings.filterwarnings(action='ignore')

from fcn import fcn8s
from pspnet import pspnet
from tqdm import tqdm 
import os
from PIL import Image

conf = {
    'model' : 'fcn8s',
    'gpu': 4,
    'dataset_dir': '/mnt/sqnap1/saugupt/public_datasets/PascalVoc2012/test/VOCdevkit/VOC2012/',
    'load_model_dir': "./experiments/fcn_cross_entropy_train_vgg16_init_weighted_loss/models",
	'load_model': "FCN8_vgg16_bestModel.net",
	'exp_dir': "fcn_cross_entropy_train_vgg16_init_weighted_loss"
}

conf['load_model_path'] = os.path.join(conf['load_model_dir'], conf['load_model'])
conf['save_path'] = './experiments/{}'.format(conf['exp_dir'])
conf['dump_path'] = os.path.join(conf['save_path'], "test_imgs/")
conf['result_path'] = os.path.join(conf['save_path'], "results/VOC2012/Segmentation/comp5_test_cls/")

os.makedirs(conf['save_path'], exist_ok=True)
os.makedirs(conf['dump_path'], exist_ok=True)
os.makedirs(conf['result_path'], exist_ok=True)

data_transform1 = None
augs = None

dst = PascalVOCLoader(root=conf['dataset_dir'], is_transform=False, transform = data_transform1, augmentations=augs, img_norm=False, split = 'test')

# img = dst[2]  
# plt.figure(0, figsize=(10, 5))
# plt.imshow(img.data.numpy().transpose([1, 2, 0]))
# plt.show()

test_loader = torch.utils.data.DataLoader(dst, batch_size=1)
# dataset_size = len(dst)
# indices = list(range(dataset_size))
# test_sampler = SequentialSampler(indices)

if conf['model'] == 'fcn8s':
	model = fcn8s(n_classes=21)
elif conf['model'] == 'pspnet':
	model = pspnet(version="pascal").float()
else:
	print("Model not specified")
	exit()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(conf['gpu'])
print("Device: ", device)

state_dict = torch.load(conf['load_model_path'], map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)


def show_prediction(img, pred, imgname, dst, save=True):
	# print(img.size(), pred.size())
	plt.figure(0, figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.imshow(img.squeeze().cpu().detach().data.numpy().transpose([1, 2, 0]))

	plt.subplot(1, 2, 2)
	plt.imshow(dst.decode_segmap(pred.squeeze().cpu().detach().data.numpy()))

	if save:
		plt.savefig(imgname)
	# plt.show()
	plt.close()

model.eval()

nbatches = len(test_loader)
pbar = tqdm(total = nbatches)

with torch.no_grad():
	for batch_idx, sample in enumerate(test_loader):
		data= sample.float().to(device)
		output = model(data)
		out_pred = output.data.max(1)[1].to(device)
		show_prediction(data, out_pred, '{}/img_{}.png'.format(conf['dump_path'], batch_idx), dst)
		pred = out_pred.squeeze().cpu().detach().data.numpy()
		pred = Image.fromarray((pred).astype(np.uint8))
		pred.save('{}/{}.png'.format(conf['result_path'],dst.files['test'][batch_idx]))
		# plt.imsave(fname='{}/{}'.format(conf['result_path'],dst.files['test'][batch_idx]), arr=out_pred.squeeze().cpu().detach().data.numpy() )
		pbar.update(1)




