import glob
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import Dataset
random.seed(0)


class SiameseDataset(Dataset):

    def __init__(self, data_root, negative_weighting=0.5, samples_to_use=1):
        self.negative_weighting = negative_weighting
        self.off_dir = os.path.join(data_root, 'off')
        self.on_images = sorted(glob.glob(os.path.join(data_root, 'on', '*.png')))
        for on_file in self.on_images:
            assert os.path.isfile(os.path.join(self.off_dir, os.path.basename(on_file)))

        temp_img = cv2.imread(on_file)
        H, W, _ = temp_img.shape

        # Shuffle so we can pull samples_to_use # of random samples from beginning of the list
        random.shuffle(self.on_images)

        assert(samples_to_use <= 1)
        self.length = int(len(self.on_images) * samples_to_use)
        print('{} total images to train over (negatives ratio: {})'.format(self.length, negative_weighting))

        self.transform = A.Compose([
            A.RandomResizedCrop(height=H, width=W, scale=(0.75, 1.5), ratio=(1, 1), interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), 
            ToTensorV2(),], 
        additional_targets={'imageOff':'image'})

    def __getitem__(self, index):
        img_on_path = self.on_images[index]
        img_off_path = img_on_path.replace('/on/', '/off/')     

        assert os.path.isfile(img_on_path)
        assert os.path.isfile(img_off_path)

        img_on = cv2.imread(img_on_path, 0) / 255
        img_off = cv2.imread(img_off_path, 0) / 255
        
        # No negative sampling here; do it in main training file
        transformed = self.transform(image=img_on, imageOff=img_off)
        img_on = transformed['image']
        img_off = transformed['imageOff']
        
        # Get the mean and std from training set
        img_on = (img_on - 0.49) / 0.135
        img_off = (img_off - 0.44) / 0.12

        img_on = img_on.float()
        img_off = img_off.float()

        return img_on, img_off

    def __len__(self):
        return self.length
