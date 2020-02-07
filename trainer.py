import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, posenet
import torch.utils.model_zoo as model_zoo

import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Posenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
          default=0, type=int)
    parser.add_argument('--epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=25, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.000001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--vdataset', dest='vdataset', help='Validation dataset type.', default='AFLW2000', type=str)
    parser.add_argument('--vdata_dir', dest='vdata_dir', help='Directory path for validation.',
          default='', type=str)
    parser.add_argument('--vfilename_list', dest='vfilename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=1, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--max_angle', dest='max_angle', help='Max angle.',
          default=99, type=int)
    parser.add_argument('--bin_angle', dest='bin_angle', help='Bin angle.',
          default=3, type=int)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    max_angle = args.max_angle
    angles_per_bin = args.bin_angle
    bins = (max_angle * 2) // angles_per_bin

    model = posenet.PoseNet(angles_per_bin, max_angle)

    #'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    #'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    #'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    #'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    #'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    #'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    #'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    vtransformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_dir = os.path.join('Data\\RAW', args.data_dir)
    vdata_dir = os.path.join('Data\\RAW', args.vdata_dir)

    if args.dataset == 'Pose_300W_Training':
        pose_dataset = datasets.Pose_300W_Training(data_dir, args.filename_list, transformations, max_angle=max_angle, bin_angle=angles_per_bin)
    elif args.dataset == 'Pose_300W_PNP_Training':
        pose_dataset = datasets.Pose_300W_PNP_Training(data_dir, args.filename_list, transformations, max_angle=max_angle, bin_angle=angles_per_bin)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    if args.vdataset == 'AFLW2000':
        validate_pose_dataset = datasets.AFLW2000(vdata_dir, args.vfilename_list, vtransformations)
    else:
        print('Error: not a valid validation dataset name')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    validation_loader = torch.utils.data.DataLoader(dataset=validate_pose_dataset,
                                                    batch_size=1,
                                                    num_workers=2)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = range(bins)
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                   lr = args.lr)

    print('Ready to train network.')
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * angles_per_bin - max_angle
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * angles_per_bin - max_angle
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * angles_per_bin - max_angle

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.ones(1).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, loss_yaw.item(), loss_pitch.item(), loss_roll.item()))

        model.eval()
        total = 0
        yaw_error = .0
        pitch_error = .0
        roll_error = .0

        for i, (images, labels, cont_labels, name) in enumerate(validation_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)

            label_yaw = cont_labels[:,0].float()
            label_pitch = cont_labels[:,1].float()
            label_roll = cont_labels[:,2].float()

            yaw, pitch, roll = model(images)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * angles_per_bin - max_angle
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * angles_per_bin - max_angle
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * angles_per_bin - max_angle

            # Mean absolute error
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

		print('Test error in degrees of the model on the ' + str(total) +
		' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (yaw_error / total,
		pitch_error / total, roll_error / total, (yaw_error + pitch_error + roll_error) / (3.0 * total)))

        model.train(True)

        # Save models at numbered epochs.
        if epoch % 1 == 0:
            print('Taking snapshot...')
            torch.save(model.state_dict(), 
            'output/snapshots/%s_epoch_%i_loss_%.3f_%.3f_%.3f.pkl' % (args.output_string, epoch+1, yaw_error / total, pitch_error / total, roll_error / total))
