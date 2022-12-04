import math
import os

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class KorniaDoG(nn.Module):

    def __init__(self, scale_pyramid=kornia.geometry.transform.pyramid.ScalePyramid()):
        super(KorniaDoG, self).__init__()
        self.scale_pyramid = scale_pyramid

    def forward(self, x):
        B, C, H, W = x.shape
        pyramids, sigmas, dists = self.scale_pyramid(x)
        out = []
        sg = []
        di = []
        for pyr, sigma, dist in zip(pyramids, sigmas, dists):
            DoG = pyr[:,:,1:,:,:] - pyr[:,:,:-1,:,:]
            dog = DoG.squeeze(dim=1)
            # dog = F.upsample(DoG.squeeze(dim=2), size=(2*H,2*W), mode='bilinear')
            out.append(dog)
            sg.append(sigma[:,1:])
            di.append(dist[:,1:])
        return out, sg, di, pyramids


class KorniaDoGScalePyr(nn.Module):

    def __init__(self, scale_pyramid=kornia.geometry.transform.pyramid.ScalePyramid()):
        super(KorniaDoGScalePyr, self).__init__()
        self.scale_pyramid = scale_pyramid

    def forward(self, x):
        B, C, H, W = x.shape
        pyramids, sigmas, dists = self.scale_pyramid(x)
        out = []
        sg = []
        di = []
        for pyr, sigma, dist in zip(pyramids, sigmas, dists):
            DoG = pyr[:,1:,:,:,:] - pyr[:,:-1,:,:,:]
            out.append(DoG.contiguous())
            sg.append(sigma[:,1:].contiguous())
            di.append(dist[:,1:])
        return out, sg, di