import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.neg_dataset import SiameseDataset
from model.correlator import Correlator
from model.unet import UNet
from utils.helper import (inference_img, make_sure_path_exists,
                          write_tensorboard)

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

mixed_precision = False
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

train_metric_labels = ['train_loss']
test_metric_labels = ['test_loss']

def eval(model, device, optimizer, scheduler, correlate, criterion, dataloader, train, epoch):
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in pbar:
            input1, input2, labels = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            labels = labels.float()

            # Feed through transformation model
            output1 = model(input1)
            output2 = model(input2)
            
            corr_score = correlate(output1, output2).squeeze()
            loss = criterion(corr_score, labels)

            if train:
                # Compute gradient
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # print statistics
            total_loss += loss.item()

            pbar_label = 'Train' if train else 'Val'
            s = ('{} Epoch {} | Batch {} | loss: {:3f}').format(pbar_label, epoch, i, loss.item())
            pbar.set_description(s)

        if len(dataloader) == 0:
            return np.zeros(1)

        return np.array([total_loss]) / len(dataloader)


def train(opt, weights_folder):

    writer = SummaryWriter(os.path.join('runs', opt.exp_name))

    # Load datasets
    trainset = SiameseDataset(data_root=opt.training_data_dir, negative_weighting=opt.negative_weighting_train, samples_to_use=opt.train_proportion)
    trainloader = utils.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    testset = SiameseDataset(data_root=opt.validation_data_dir, negative_weighting=0.5, samples_to_use=opt.train_proportion)
    testloader = utils.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # Set gpu stuff
    devices = opt.device.split(',')
    device = torch.device("cuda:" + devices[0] if torch.cuda.is_available() else "cpu")
    print('Using gpu')

    # Load model
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.to(device)

    correlate = Correlator(device=device)
    criterion = nn.MSELoss()
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995, last_epoch=-1)

    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    test_best_loss = np.inf
    train_best_loss = np.inf
    N_epochs = opt.epochs

    for epoch in range(1, N_epochs + 1):  # loop over the dataset multiple times

        model.train()
        train_metrics = eval(model, device, optimizer, scheduler, correlate, criterion, trainloader, train=True, epoch=epoch)
        write_tensorboard(writer, train_metric_labels, train_metrics, epoch)

        # Evaluate on test set
        with torch.no_grad():

            model.eval()
            test_metrics = eval(model, device, optimizer, scheduler, correlate, criterion, testloader, train=False, epoch=epoch)

            total_train_loss = train_metrics[0]
            total_test_loss = test_metrics[0]
            if total_test_loss <= test_best_loss:
                test_best_loss = total_test_loss
                torch.save(model.state_dict(), os.path.join(weights_folder, 'best_test_weights.pt'))

            if total_train_loss <= train_best_loss:
                train_best_loss = total_train_loss
                torch.save(model.state_dict(), os.path.join(weights_folder, 'best_train_weights.pt'))

            write_tensorboard(writer, test_metric_labels, test_metrics, epoch)
    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)   
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--exp_name')
    parser.add_argument('--training_data_dir', help='directory level containing "on" and "off" folders of images')
    parser.add_argument('--validation_data_dir', help='directory level containing "on" and "off" folders of images')

    parser.add_argument('--negative_weighting_train', type=float, default=0.5)
    parser.add_argument('--train_proportion', type=float, default=1, help='ratio of dataset to use during training')

    opt = parser.parse_args()
    print(opt)

    weights_folder = os.path.join('experiments', '{}'.format(opt.exp_name), 'weights')
    make_sure_path_exists(weights_folder)
    
    train(opt, weights_folder)

