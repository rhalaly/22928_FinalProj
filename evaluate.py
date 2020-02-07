import sys, os, argparse

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

import datasets, posenet, utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)
    parser.add_argument('--max_angle', dest='max_angle', help='Max angle.',
          default=99, type=int)
    parser.add_argument('--bin_angle', dest='bin_angle', help='Bin angle.',
          default=3, type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    max_angle = args.max_angle
    angles_per_bin = args.bin_angle
    bins = (max_angle * 2) // angles_per_bin

    # ResNet50 structure
    model = posenet.PoseNet(angles_per_bin, max_angle)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_dir = os.path.join('Data\\RAW', args.data_dir)

    if args.dataset == 'Pose_300W_LP_Training':
        pose_dataset = datasets.Pose_300W_LP_Training(data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=1,
                                               num_workers=2)

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = range(bins)
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
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

        # Save first image in batch with pose cube or axis.
        if args.save_viz:
            name = name[0]
            if args.dataset == 'BIWI':
                cv2_img = cv2.imread(os.path.join(data_dir, name + '_rgb.png'))
            else:
                cv2_img = cv2.imread(os.path.join(data_dir, name + '.jpg'))

            predicted = 'y %.2f, p %.2f, r %.2f' % (yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item())
            actual = 'y %.2f, p %.2f, r %.2f'  % (label_yaw.item(), label_pitch.item(), label_roll.item())
            error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)), torch.sum(torch.abs(pitch_predicted - label_pitch)), torch.sum(torch.abs(roll_predicted - label_roll)))
            cv2.putText(cv2_img, predicted, (30, cv2_img.shape[0] - 90), fontFace=1, fontScale=1, color=(0,255,0), thickness=2)
            cv2.putText(cv2_img, actual, (30, cv2_img.shape[0] - 60), fontFace=1, fontScale=1, color=(0,255,255), thickness=2)
            cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0] - 30), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)

            utils.draw_axis_rads(cv2_img, *np.radians([yaw_predicted[0], pitch_predicted[0], roll_predicted[0]]))

            utils.draw_axis_rads(cv2_img, *np.radians([label_yaw[0], label_pitch[0], label_roll[0]]), cx=(255,255,0), cy=(255,0,255), cz=(0, 255, 255))

            cv2.imwrite(os.path.join('output\\images', name + '.jpg'), cv2_img)

    print('Test error in degrees of the model on the ' + str(total) +
    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (yaw_error / total,
    pitch_error / total, roll_error / total, (yaw_error + pitch_error + roll_error) / (3.0 * total)))
