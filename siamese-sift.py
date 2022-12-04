import argparse
import math
import os
import random

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.correlator import Correlator
from utils.helper import make_sure_path_exists, write_tensorboard
from model.kornia_dog import KorniaDoG
from model.kornia_sift import KorniaSift
from dataset.neg_sift_dataset import SiameseDataset
from model.unet import UNet


def laf_from_opencv_kpts(kpts, mrSize=6.0, device=torch.device('cpu')):
    N = len(kpts)
    xy = torch.tensor([(x.pt[0], x.pt[1]) for x in kpts ], device=device, dtype=torch.float).view(1, N, 2)
    scales = torch.tensor([(mrSize * x.size) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1, 1)
    angles = torch.tensor([(x.angle) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1)
    laf = kornia.feature.laf_from_center_scale_ori(xy, scales, -angles)
    return laf

def repeatListToLengthN(lst, N):
    repeat = math.ceil(N / len(lst))
    lst = lst*repeat
    lst = lst[:N]

    return lst

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)
random.seed(0)

mixed_precision = False
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

train_metric_labels = ['train_loss', 'train_top_pyramid_loss', 'train_top_sift_loss']
test_metric_labels = ['test_loss', 'test_top_pyramid_loss', 'test_top_sift_loss']

def eval(model, optimizer, scheduler, device, dog, sift, criterion, correlate, dataloader, cv2_sift, numFeatures, train=True, epoch=0):
    total_loss = 0
    total_top_pyramid_loss = 0
    total_top_sift_loss = 0
    
    optimizer.zero_grad()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in pbar:
        
        top_sift_loss, top_pyramid_loss = torch.zeros(2).to(device)
        input1, input2 = data[0].to(device), data[0].to(device)

        # Feed through transformation model
        output1 = model(input1)
        output2 = model(input2)
                    
        # Top loss
        pyr_nms_output1, _, _, sp1 = dog(output1)
        pyr_nms_output2, _, _, sp2 = dog(output2) 
        num_octaves = len(pyr_nms_output1)

        rsp1 = []
        rsp2 = []

        B, _, _, _ = output1.shape
        targets = torch.ones(opt.subsamples*B).to(device)
        for j in range(opt.subsamples):    
            octave_random = random.randint(0, num_octaves - 1)
            _, num_layers, H, W = pyr_nms_output1[octave_random].shape

            layer_random = random.randint(0, num_layers - 1)
            layer1 = pyr_nms_output1[octave_random][:, layer_random]
            layer2 = pyr_nms_output2[octave_random][:, layer_random]

            ub_h = H - opt.crop_width
            ub_w = W - opt.crop_width

            r_on = random.randint(0, ub_h - 1)
            c_on = random.randint(0, ub_w - 1)

            r_off = r_on
            c_off = c_on

            if random.random() < 0.5:
                while c_off == c_on and r_on == r_off:
                    c_off = random.randint(0, ub_w - 1)
                    r_off = random.randint(0, ub_h - 1)

                targets[B*j:B*j+B] = 0.0

            crop1 = layer1[:, r_on:r_on + opt.crop_width, c_on:c_on + opt.crop_width].contiguous().view(B, 1, opt.crop_width, opt.crop_width)
            crop2 = layer2[:, r_off:r_off + opt.crop_width, c_off:c_off + opt.crop_width].contiguous().view(B, 1, opt.crop_width, opt.crop_width)
            
            rsp1.append(crop1)
            rsp2.append(crop2)

        rsp1_tensor = torch.cat(rsp1, dim=0)
        rsp2_tensor = torch.cat(rsp2, dim=0)

        top_pyramid_loss = criterion(correlate(rsp1_tensor, rsp2_tensor).squeeze(), targets.float())

        desc1, desc2, laf1, laf2 = None, None, None, None

        # Open CV detect lafs
        cpu_output1 = (output1*255).squeeze(dim=1).byte().cpu().detach().numpy()
        cpu_output2 = (output2*255).squeeze(dim=1).byte().cpu().detach().numpy()

        laf1_vec = []
        laf2_vec = []
        for b in range(B):
            o1, o2 = cpu_output1[b], cpu_output2[b]

            kp1 = cv2_sift.detect(o1)
            kp2 = cv2_sift.detect(o2)
            
            # img1 = cv2.drawKeypoints(o2, kp2, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255, 0, 0))
            # plt.imshow(img1)
            # plt.show()
            if len(kp1) == 0 or len(kp2) == 0:
                continue

            kp1 = repeatListToLengthN(kp1, numFeatures)
            kp2 = repeatListToLengthN(kp2, numFeatures)

            torch_kp1 = laf_from_opencv_kpts(kp1, device=device)
            torch_kp2 = laf_from_opencv_kpts(kp2, device=device)

            laf1_vec.append(torch_kp1)
            laf2_vec.append(torch_kp2)

        descriptor_positive, descriptor_negative, descriptor_match_map = torch.zeros(3, device=device)
        if len(laf1_vec) != 0 and len(laf2_vec) != 0:
            laf1 = torch.cat(laf1_vec, dim=0)
            laf2 = torch.cat(laf2_vec, dim=0)

            desc1, _ = sift(output1, laf=laf1)
            desc2, _ = sift(output2, laf=laf2)

            # x1, y1 = kornia.feature.laf.get_laf_pts_to_draw(laf1, 0)
            # x2, y2 = kornia.feature.laf.get_laf_pts_to_draw(laf2, 0)
            # _, N = x1.shape

            # B, C, H, W = output1.shape
            # output = torch.cat([output1, output2], dim=3).squeeze()
            # plt.imshow(output.cpu().detach().numpy(), cmap='gray')
            # for i in range(N):
            #     plt.plot(x1[:,i], y1[:,i])
            #     plt.plot(W + x2[:,i], y2[:,i])
            # plt.show()

            ### LAF operations
            laf1_scales = kornia.feature.get_laf_scale(laf1).squeeze(dim=2)
            laf2_scales = kornia.feature.get_laf_scale(laf2).squeeze(dim=2)
            assert(torch.isnan(laf1_scales).any() == False)
            assert(torch.isnan(laf2_scales).any() == False)
            laf1_centers = kornia.feature.get_laf_center(laf1)
            laf2_centers = kornia.feature.get_laf_center(laf2)
            assert(torch.isnan(laf1_centers).any() == False)
            assert(torch.isnan(laf2_centers).any() == False)
            scale_dist = torch.cdist(laf1_scales, laf2_scales)
            center_dist = torch.cdist(laf1_centers, laf2_centers)

            assert(torch.isnan(scale_dist).any() == False)
            assert(torch.isnan(center_dist).any() == False)

            scale_matchmap = (scale_dist < 2)           # Match only if keypoint scales are similar
            center_dist_thresh_map = (center_dist <= 5) # Match only if keypoints are very close
            descriptor_match_map = (center_dist_thresh_map & scale_matchmap) 

            assert(torch.isnan(scale_matchmap).any() == False)
            assert(torch.isnan(center_dist_thresh_map).any() == False)
            assert(torch.isnan(descriptor_match_map).any() == False)

            descriptor_dist = torch.cdist(desc1, desc2)

            assert(torch.isnan(descriptor_dist).any() == False)

            if len(descriptor_dist[descriptor_match_map]) != 0:
                descriptor_positive = descriptor_dist[descriptor_match_map].mean()

            if len(descriptor_dist[~descriptor_match_map]) != 0:
                descriptor_negative = descriptor_dist[~descriptor_match_map].mean()

            assert(torch.isnan(descriptor_positive).any() == False)
            assert(torch.isnan(descriptor_negative).any() == False)

            top_sift_loss = descriptor_positive + F.relu(2 - descriptor_negative)

        loss = opt.gamma*top_sift_loss + opt.zeta*top_pyramid_loss

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
        total_top_pyramid_loss += top_pyramid_loss.item()
        total_top_sift_loss += top_sift_loss.item()

        pbar_label = 'Train' if train else 'Test'
        s = ('{} Epoch {} | Batch {} | loss: {:3f} | pyramid loss: {:3f}, descriptor loss: {:3f}').format(
            pbar_label,
            epoch,
            i, 
            loss.item(), 
            top_pyramid_loss.item(),
            top_sift_loss.item(),
        )
        pbar.set_description(s)

    if len(dataloader) == 0:
        return np.zeros(3)
    return np.array([total_loss, total_top_pyramid_loss, total_top_sift_loss]) / len(dataloader)


def train(opt, weights_folder):
    writer = SummaryWriter(os.path.join('runs', opt.exp_name))

    # Load datasets
    trainset = SiameseDataset(opt.training_data_dir, negative_weighting=0.5, samples_to_use=opt.train_proportion)
    testset = SiameseDataset(opt.validation_data_dir, negative_weighting=0.5, samples_to_use=opt.train_proportion)

    trainloader = utils.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    testloader = utils.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    # Set gpu stuff
    devices = opt.device.split(',')
    
    device = torch.device("cuda:" + devices[0] if torch.cuda.is_available() else "cpu")
    print('Using gpu')

    # Load model
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.to(device)

    scale_pyr = kornia.geometry.ScalePyramid(
        n_levels=3,
        init_sigma=1.6,
        min_size=80,
        double_image=True)

    numFeatures = 500

    correlate = Correlator(device=device)
    dog = KorniaDoG(scale_pyramid=scale_pyr)
    cv2_sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, nOctaveLayers=3, nfeatures=numFeatures)
    sift = KorniaSift().to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    test_best_loss = np.inf
    train_best_loss = np.inf
    N_epochs = opt.epochs
    for epoch in range(1, N_epochs + 1):  # loop over the dataset multiple times
        model.train()

        train_metrics = eval(
            model, optimizer, scheduler, 
            device, 
            dog, sift, criterion, correlate, 
            trainloader, cv2_sift, numFeatures, train=True, epoch=epoch
        )

        write_tensorboard(writer, train_metric_labels, train_metrics, epoch)

        # Evaluate on test set
        with torch.no_grad():
            model.eval()

            test_metrics = eval(
                model, optimizer, scheduler, 
                device, 
                dog, sift, criterion, correlate, 
                testloader, cv2_sift, numFeatures, train=False, epoch=epoch
            )

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
    parser.add_argument('--batch-size', type=int, default=2)   
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--zeta', type=float, default=10, help='descriptor loss weighting')
    parser.add_argument('--gamma', type=float, default=1, help='detector loss weighting')

    parser.add_argument('--exp_name')
    parser.add_argument('--training_data_dir', help='directory level containing "on" and "off" folders of images')
    parser.add_argument('--validation_data_dir', help='directory level containing "on" and "off" folders of images')

    parser.add_argument('--train_proportion', type=float, default=1, help='ratio of dataset to use during training')
    parser.add_argument('--negative_weighting_train', type=float, default=0.5)
    parser.add_argument('--subsamples', type=int, default=100)
    parser.add_argument('--crop_width', type=int, default=64)

    opt = parser.parse_args()
    print(opt)

    weights_folder = os.path.join('experiments', '{}'.format(opt.exp_name), 'weights')
    make_sure_path_exists(weights_folder)
    
    train(opt, weights_folder)

