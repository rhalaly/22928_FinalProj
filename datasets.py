import os
import numpy as np
import cv2
import pandas as pd
import glob
import math

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    if os.path.isfile(file_path):
        with open(file_path) as f:
            lines = f.read().splitlines()
        return lines

    files = glob.glob(os.path.join(file_path, '**', '*.png'), recursive=True)
    files.extend(glob.glob(os.path.join(file_path, '**', '*.jpg'), recursive=True))

    files = map(lambda fp: fp.replace('.png', '').replace('.jpg', ''), files)
    return files

class Pose_300W_Training(Dataset):
    # 300W-LP dataset with random downsampling
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB', max_angle=99, bin_angle=3):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.max_angle = max_angle
        self.bin_angle = bin_angle

        filename_list = get_list_from_filenames(os.path.join(data_dir, filename_path) if filename_path != '' else data_dir)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
        self.bins = np.array(range(-max_angle, max_angle + bin_angle, bin_angle))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
    
        ds = 1 + np.random.randint(0,4) * 5
        original_size = img.size
        img = img.resize((img.size[0] // ds, img.size[1] // ds), resample=Image.NEAREST)
        img = img.resize((original_size[0], original_size[1]), resample=Image.NEAREST)

        # Rotate
        angle = np.random.uniform(-50, 50)
        R = utils.create_rotation_matrix(yaw, pitch, roll)
        R = np.matmul(R, utils.create_rotation_matrix(0, 0, -angle))
        yaw_r, pitch_r, roll_r = utils.rotation_matrix_to_euler_angles(R)
        if np.abs(yaw_r) <= self.max_angle and np.abs(pitch_r) <= self.max_angle and np.abs(roll_r) <= self.max_angle:
            img = img.rotate(angle, resample=Image.NEAREST)
            #utils.draw_axis_pil(img, yaw_r, pitch_r, roll_r).show()
            yaw, pitch, roll = yaw_r, pitch_r, roll_r
            #exit()

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Change contrast?
        rnd = np.random.random_sample()
        if rnd < 0.35:
            if np.random.random_sample() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.1, 0.9))
            else:
                img = ImageEnhance.Contrast(img).enhance(np.random.uniform(1.1, 1.9))

        # Change brightness?
        rnd = np.random.random_sample()
        if rnd < 0.35:
            if np.random.random_sample() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.15, 0.9))
            else:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(1.1, 1.85))

        # Change sharpness?
        rnd = np.random.random_sample()
        if rnd < 0.1:
            img = ImageEnhance.Sharpness(img).enhance(np.random.uniform(0.1, 1.9))

        # Grayscale?
        rnd = np.random.random_sample()
        if rnd < 0.35:
            img = img.convert('L')
            img = img.convert(mode='RGB')

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        binned_pose = np.digitize([yaw, pitch, roll], self.bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        return self.length

class Pose_300W_PNP_Training(Dataset):
    # 300W-LP with PNP tagging
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.pts', image_mode='RGB', max_angle=99, bin_angle=3):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.max_angle = max_angle
        self.bin_angle = bin_angle

        filename_list = get_list_from_filenames(os.path.join(data_dir, filename_path) if filename_path != '' else data_dir)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
        self.bins = np.array(range(-max_angle, max_angle + bin_angle, bin_angle))
        self.head3d = utils.get_head_model('Data\\RAW\\model3D_aug_-00_00_01.mat')
        self.camera = utils.get_camera_matrix('Data\\RAW\\model3D_aug_-00_00_01.mat')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index]))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index].replace('.jpg', '').replace('.png', '') + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_pts(mat_path)
        x_min = min(pt2d[:,0])
        y_min = min(pt2d[:,1])
        x_max = max(pt2d[:,0])
        y_max = max(pt2d[:,1])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Rotate
        angle = np.random.uniform(-30, 30)
        img = img.rotate(-angle, resample=Image.NEAREST)

        pts = []

        for pt in pt2d:
            x = pt[0] - x_min
            y = pt[1] - y_min

            p = utils.rotate2d((img.size[0] // 2, img.size[1] // 2), (x, y), np.radians(angle))
            pts.append(p)

        ptsf = np.array(pts, dtype="double")
        (_, rotation_vector, _) = cv2.solvePnP(self.head3d, ptsf, self.camera, None, flags=cv2.SOLVEPNP_ITERATIVE)

        R, _ = cv2.Rodrigues(rotation_vector)
        yaw, pitch, roll = utils.rotation_matrix_to_euler_angles(R)

        if abs(roll) > 99:
            if roll > 0:
                roll = 180 - roll
            else:
                roll = 180 + roll

        # Bin values
        binned_pose = np.digitize([yaw, pitch, roll], self.bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        return self.length

class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB', max_angle=99, bin_angle=3):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(os.path.join(data_dir, filename_path) if filename_path != '' else data_dir)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
        self.bins = np.array(range(-max_angle, max_angle + bin_angle, bin_angle))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], self.bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length