import argparse
import os

import torch
import torch.utils.data as utils
import torchvision
from tqdm import tqdm

from dataset.inference_dataset import InferenceDataset
from model.unet import UNet

def createDirectory(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)

def inference(opt):

    batch_size = opt.batch_size
    devices = opt.device.split(',')

    dataset = InferenceDataset(opt.data_dir, mean=opt.mean, std=opt.std)
    loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)

    # Load model
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(torch.load(opt.weights_path))

    # Set gpu stuff
    device = torch.device("cuda:" + devices[0] if torch.cuda.is_available() else "cpu")
    print("using cuda:" + devices[0])
    model.to(device)

    with torch.no_grad():
        model.eval()

        pbar = tqdm(enumerate(loader), total=len(loader))
        for i, data in pbar:
            input1, img_name = data[0].to(device), data[1]            
            output1 = model(input1)

            for k in range(output1.size(0)):
                img = output1[k, :, :, :]   
                torchvision.utils.save_image(img, os.path.join(opt.output_dir, img_name[k]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--weights_path')

    parser.add_argument('--batch-size', type=int, default=1)   
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--num_workers', type=int, default=1)
    
    parser.add_argument('--mean', type=float, default=0.4)
    parser.add_argument('--std', type=float, default=0.12)
    
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.output_dir, exist_ok=True)
    inference(opt)

