import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import functools
from torchvision import transforms
import time
from data_loader_pascal import PascalVOCLoader
from loss import cross_entropy2d, dice_loss, iou, accuracy
from img_utils import showBatchImage


def train(model, device, train_loader, optimizer, epoch, conf, loss_fn, num_classes):
    model.train()
    print_int = 10
    n = print_int* conf['batch_size']
    acc = 0.0
    avg_loss = 0.0
    since = time.time()
    global_loss = 0.0

    for batch_idx, sample in enumerate(train_loader):
        data, target = sample[0].float().to(device), sample[1].float().to(device)
        optimizer.zero_grad()
        output = model(data)
        out_masks = nn.Sigmoid()(output)
        out_pred = output.data.max(1)[1].to(device)

        if len(target.shape) < 4:
            tg = torch.FloatTensor(target.size(0), num_classes, target.size(1), target.size(2)).zero_().cuda()
            tg = tg.scatter_(1, target.unsqueeze(1).long(), 1)
            target = tg

        # print(data.shape, target.shape, output.shape, out_masks.shape, target.shape)
        acc += iou(out_masks, target)/conf['batch_size']
        loss = loss_fn(input = output, target = target)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
        global_loss += loss.item()
        
        if batch_idx % print_int == 0:
            print('Train Epoch: {} [{}/{:.0f} ]\tLoss: {:.6f}\tMeanIOU: {:.3f}\tTime(in sec): {:.0f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset) * 0.9,
                  avg_loss / print_int, acc/print_int, time.time() - since))
            avg_loss = 0.0
            acc = 0.0
            since = time.time()

    f = open("./logs/train_loss.txt", "a+")
    f.write("{}\n".format(global_loss))
    f.close()

# from sklearn.metrics import confusion_matrix
import torch.nn.functional as f1


def val(model, device, test_loader, epoch, data_size, conf, loss_fn, num_classes):
    model.eval()
    test_loss = 0
    batch_loss = 0
    print_int = 10
    n = print_int* conf['batch_size']
    acc = 0.0
    global_iou = 0.0
    print("----------------------------------------------------------------")
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample[0].float().to(device), sample[1].float().to(device)
            output = model(data)
            out_masks = nn.Sigmoid()(output)
            out_pred = output.data.max(1)[1].to(device)
            label = target

            if len(target.shape) < 4:
                tg = torch.FloatTensor(target.size(0), num_classes, target.size(1), target.size(2)).zero_().cuda()
                tg = tg.scatter_(1, target.unsqueeze(1).long(), 1)
                target = tg

            acc += iou(out_masks, target)/conf['batch_size']
            global_iou += iou(out_masks, target)

            loss = loss_fn(input = output, target = target)

            test_loss += loss.item()
            batch_loss += loss.item()
            
            if batch_idx %10 == 0:
                print('Validation Epoch: {} [{}/{:.0f} ]\tLoss:{:.4f}\tMeanIOU:{:.3f}'.format(epoch, batch_idx * len(data), data_size,  batch_loss/print_int, acc/print_int))
                batch_loss = 0
                for i in range(conf['batch_size']):
                    showBatchImage(data, label, output, './img_dmp/img_{}_{}.png'.format(epoch, batch_idx))
                acc = 0.0                
                        
    test_loss /= len(test_loader)
    
    f = open("./logs/val_loss.txt", "a+")
    f.write("{}\n".format(test_loss))
    f.close()

    print('\nTest set: Average loss: {:.4f}\t GlobalIOU: {:.2f}\n'.format(test_loss, global_iou/data_size))
    
def test(model, device, test_loader, data_size, conf):
    model.eval()
    print("----------------------------------------------------------------")
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data = sample['image']
            data = data.float().to(device)
            output = model(data)
            out_masks = nn.Sigmoid()(output)
            for i in range(4):
                show_sample3(torch.squeeze(data[i], 0), out_masks[i].repeat(3, 1, 1), "img_%d_%d"%(batch_idx, i))
                
    return out_masks
    

