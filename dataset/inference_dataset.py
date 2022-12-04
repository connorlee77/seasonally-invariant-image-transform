import os
import tqdm
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class InferenceDataset(Dataset):

    def __init__(self, data_dir, mean=0.5, std=0.1):

        self.mean = mean
        self.std = std

        self.images = glob.glob(os.path.join(data_dir, '*.png'))
        self.length = len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        
        img = Image.open(img_path)     
        img = transforms.ToTensor() (img.convert('L'))
        img = (img - self.mean) / self.std
        return img, os.path.basename(img_path)

    def __len__(self):
        return self.length

