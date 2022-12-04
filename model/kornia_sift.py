import math
import os

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.kornia_dog import KorniaDoGScalePyr


class KorniaSift(nn.Module):

    def __init__(self, num_features=500, minima_are_also_good=False, scale_pyr=None, 
        nms=None, resp=None, ori_module=None):

        super(KorniaSift, self).__init__()
        self.num_features = num_features
        self.minima_are_also_good = minima_are_also_good

        self.resp = resp
        self.nms = nms
        self.scale_pyr = scale_pyr

        self.detect = kornia.feature.ScaleSpaceDetector(
            num_features=self.num_features, 
            minima_are_also_good=self.minima_are_also_good,
            resp_module=resp, 
            nms_module=nms,
            scale_pyr_module=self.scale_pyr, 
            ori_module=ori_module)
        
        self.get_descriptor = kornia.feature.SIFTDescriptor(patch_size=32)

    def forward(self, x, laf=None):
        '''
            input: batch of images (B, C, H, W)
            output: batch of descriptors (B, N_Descriptors, 128)
        '''
        B, C, H, W = x.shape

        if laf is None:
            laf, resp = self.detect(x)

        patches = kornia.feature.extract_patches_from_pyramid(x, laf, PS=32)
        PB, N_KEYPOINTS, PC, PH, PW = patches.shape
        
        assert(PC == 1)

        patches = patches.view(PB*N_KEYPOINTS, PC, PH, PW)
        desc = self.get_descriptor(patches)

        # SIFT should have 128 features in descriptor
        desc = desc.view(PB, N_KEYPOINTS, 128)
        return desc, laf