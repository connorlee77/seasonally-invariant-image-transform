import os
import random
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
random.seed(0)

class SiameseDataset(Dataset):

    def __init__(self, data_root, negative_weighting=0.5, samples_to_use=1):
        self.negative_weighting = negative_weighting
        self.off_dir = os.path.join(data_root, 'off')
        self.on_images = sorted(glob.glob(os.path.join(data_root, 'on', '*.png')))
        for on_file in self.on_images:
            assert os.path.isfile(os.path.join(self.off_dir, os.path.basename(on_file)))

        # Shuffle so we can pull samples_to_use # of random samples from beginning of the list
        random.shuffle(self.on_images)

        assert(samples_to_use <= 1)
        self.length = int(len(self.on_images) * samples_to_use)
        print('{} total images to train over (negatives ratio: {})'.format(self.length, negative_weighting))

    def __getitem__(self, index):
        img_on_path = self.on_images[index]
        img_off_path = img_on_path.replace('/on/', '/off/')     

        assert os.path.isfile(img_on_path)
        assert os.path.isfile(img_off_path)
        
        img_on = Image.open(img_on_path)
        img_off = Image.open(img_off_path)        
        
        use_negative = random.random() < self.negative_weighting
        target = 0 if use_negative else 1
        if use_negative:
            rand_index = random.randint(0, self.__len__() - 1)
            while rand_index == index:
                rand_index = random.randint(0, self.__len__() - 1)
            
            img_off_path = self.on_images[rand_index].replace('/on/', '/off/')    
            assert os.path.isfile(img_off_path)
            img_off = Image.open(img_off_path)

        img_on = transforms.ToTensor() (img_on.convert('L'))
        img_off = transforms.ToTensor() (img_off.convert('L'))

        # Get the mean and std from training set
        img_on = (img_on - 0.49) / 0.12
        img_off = (img_off - 0.44) / 0.10

        return (img_on, img_off), target


    def __len__(self):
        return self.length


        