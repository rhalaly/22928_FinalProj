import torch
import cv2
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

import utils

class PoseNet(nn.Module):
    def __init__(self, angles_bin=3, max_angle=99, cnn=torchvision.models.resnet101):
        super(PoseNet, self).__init__()

        resnet = cnn()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        num_bins = (max_angle * 2) // angles_bin # default is -99 to +98 divided to bin per 3 degrees
        self.fc_yaw = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * torchvision.models.resnet.Bottleneck.expansion, num_bins)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class Predictor():
    def __init__(self, weights_path, gpu_id, angles_bin=3, max_angle=99, cnn=torchvision.models.resnet101):
        self.__gpu_id = gpu_id
        self.__angles_bin = angles_bin
        self.__max_angle = max_angle
        self.__model = self.__get_model(weights_path, cnn)
        self.__preprocess_input = self.__get_input_transformation()
        self.__angles_indexs = self.__create_angles()

    def predict(self, rgb_image, flip_pitch=False):
        img = self.__transform_image(rgb_image)

        angles = self.__prediction_to_deg(*self.__model(img))

        if flip_pitch:
            rvec = self.__get_rotation_vec([angles[0], -angles[1], angles[2]])
        else:
            rvec = self.__get_rotation_vec(angles)

        return rvec.reshape(3,), angles, utils.draw_axis_rads(np.copy(rgb_image), *np.radians(angles))

    def __prediction_to_deg(self, yaw_bins, pitch_bins, roll_bins):
        yaw_soft = F.softmax(yaw_bins)
        pitch_soft = F.softmax(pitch_bins)
        roll_soft = F.softmax(roll_bins)

        yaw = torch.sum(yaw_soft.data[0] * self.__angles_indexs) * self.__angles_bin - self.__max_angle
        pitch = torch.sum(pitch_soft.data[0] * self.__angles_indexs) * self.__angles_bin - self.__max_angle
        roll = torch.sum(roll_soft.data[0] * self.__angles_indexs) * self.__angles_bin - self.__max_angle

        return yaw.item(), pitch.item(), roll.item()

    def __transform_image(self, rgbImage):
        img = Image.fromarray(rgbImage)
        img = self.__preprocess_input(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(self.__gpu_id)
        return img

    def __get_model(self, weights_path, cnn):
        model = PoseNet(self.__angles_bin, self.__max_angle, cnn)
        model.load_state_dict(torch.load(weights_path))
        model.cuda(self.__gpu_id)
        model.eval()
        return model

    def __create_angles(self):
        angles = range((self.__max_angle * 2) // self.__angles_bin)
        angles = torch.FloatTensor(angles).cuda(self.__gpu_id)
        return angles

    @staticmethod
    def __get_input_transformation():
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def __get_rotation_vec(pose_degs):
        matrix =  utils.create_rotation_matrix(*pose_degs)
        rvec, _ = cv2.Rodrigues(matrix)
        return rvec

