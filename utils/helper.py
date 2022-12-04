import os, errno
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils


class Normer(object):
    
    def __call__(self,sample, epsilon=1e-7):
        sample = (sample - torch.mean(sample)) / torch.std(sample + epsilon)

        return sample 

def make_sure_path_exists(path):

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def inference_img(img_path, device, model, mean=0, std=0):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        Normer()
    ])

    img = Image.open(img_path).convert('L')
    w,h = img.size
    w = 2000 if w > 2000 else w
    h = 2000 if h > 2000 else h
    img = img.crop((0,0,w,h))

    # img = transform(img)
    img = transforms.ToTensor()(img)
    img = (img - mean) / std
    # img = (img - img.mean()) / (img.std() + 1e-8)
    
    img = img.unsqueeze(0)
    input_img = img.to(device)
    output = model(input_img) # on

    return output

def write_tensorboard(writer, labels, metrics, epoch):
    for label, metric in zip(labels, metrics):
        writer.add_scalar(label, metric, epoch)

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

def rgb2gray_batch(batch):
    # normalize using imagenet mean and std
    return torch.sum(batch, dim=1, keepdim=True) / 3.0

def pyramid_loss(p1, p2, labels, correlate, criterion):
    loss = 0

    for it in range(len(p1)):
        l1 = p1[it]
        l2 = p2[it]         

        B, C, H, W = l1.shape
        
        # Each channel is own image. 
        l1 = l1.view(B*C, 1, H, W)
        l2 = l2.view(B*C, 1, H, W)

        corr_score = correlate(l1, l2).squeeze()
        
        # Batch has been combined with channels. Repeat labels C times
        loss += criterion(corr_score, labels.repeat_interleave(C))

    return loss

def pyramid_loss_mse(p1, p2, criterion):
    loss = 0

    for it in range(len(p1)):
        l1 = p1[it]
        l2 = p2[it]        

        loss += criterion(l1, l2)
    return loss