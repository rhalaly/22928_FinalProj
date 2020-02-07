import numpy as np
import torch
import os
import scipy.io as sio
import cv2
from math import cos, sin, atan2
from PIL import ImageDraw

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def create_rotation_matrix(yaw, pitch, roll):
    phi, gamma, theta = np.radians([pitch, yaw, roll])

    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
    R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return np.matmul(R_z, np.matmul(R_y, R_x))

def rotation_matrix_to_euler_angles(R) :
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else :
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0

    return tuple(np.degrees(np.array([y, x, z])))

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def draw_axis_rads(img, yaw, pitch, roll, tdx=None, tdy=None, size=100, cx=(0,0,255), cy=(0,255,0), cz=(255,0,0)):
    yaw = -yaw

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),cx,3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),cy,3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),cz,2)

    return img

def draw_axis_pil(img, yaw, pitch, roll, size=100, cx=(0,0,255), cy=(0,255,0), cz=(255,0,0)):
    draw = ImageDraw.Draw(img)
    
    tdx, tdy = img.size
    tdx //= 2
    tdy //= 2

    R = create_rotation_matrix(-yaw, pitch, -roll)

    x, y, _ = np.matmul([size, 0, 0], R)
    draw.line((int(tdx), int(tdy), int(x + tdx), int(y + tdy)), cz, 3)

    x, y, _ = np.matmul([0, size, 0], R)
    draw.line((int(tdx), int(tdy), int(x + tdx), int(y + tdy)), cy, 3)

    x, y, _ = np.matmul([0, 0, -size], R)
    draw.line((int(tdx), int(tdy), int(x + tdx), int(y + tdy)), cx, 3)

    draw.text((30, img.size[1] - 30), "y = %0.2f, p = %0.2f, r = %0.2f" % (yaw, pitch, roll))

    del draw
    return img

def rotate2d(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

def get_camera_matrix(mat_path):
    model = sio.loadmat(mat_path)["model3D"]
    return np.asmatrix(model['outA'][0, 0], dtype='float32')

def get_head_model(mat_path):
    model = sio.loadmat(mat_path)["model3D"]
    return np.asmatrix(model['threedee'][0, 0], dtype='float32')