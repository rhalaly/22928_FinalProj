import os
import cv2
import csv
import numpy as np
from posenet import PoseNet, Predictor
import utils
import torchvision

import argparse

parser = argparse.ArgumentParser(description='Head pose estimation using the Posenet network.')
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
parser.add_argument('--posenet_type', dest='posenet_type', help='The type of PoseNet to test 0 - Regular, 1 - PnP',
        default=0, type=int)
parser.add_argument('--output_dir', dest='output_dir', help='The output directory',
        default='output\\validation\\', type=str)
parser.add_argument('--dataset_dir', dest='dataset_dir', help='The validation set directory',
        default='Data\\RAW\\valid_set2\\', type=str)
parser.add_argument('--models_dir', dest='models_dir', help='The models directory',
        default='models\\', type=str)

args = parser.parse_args()

print('Loading models')

if args.posenet_type == 0:
    predictor1 = Predictor(os.path.join(args.models_dir, 'posenet101.pkl'), args.gpu_id)
    predictor2 = Predictor(os.path.join(args.models_dir, 'posenet101_less_robust.pkl'), args.gpu_id)
    predictor3 = Predictor(os.path.join(args.models_dir, 'posenet101_no_downsampling.pkl'), args.gpu_id)
    predictor4 = Predictor(os.path.join(args.models_dir, 'posenet101_noaug.pkl'), args.gpu_id)
elif args.posenet_type == 1:
    predictor1 = Predictor(os.path.join(args.models_dir, 'posenet101_pnp.pkl'), args.gpu_id)
    predictor2 = Predictor(os.path.join(args.models_dir, 'posenet101_pnp2.pkl'), args.gpu_id)
    predictor3 = Predictor(os.path.join(args.models_dir, 'posenet101_pnp3.pkl'), args.gpu_id)
    predictor4 = Predictor(os.path.join(args.models_dir, 'posenet101_pnp4.pkl'), args.gpu_id)
else:
    print('Wrong PoseNet type - choose 0 for PoseNet or 1 for PoseNet_pnp')
    exit()

def load(dots):
    return np.array([tuple(map(float, dot.strip('][').split())) for dot in dots])

thetas = []

with open(args.dataset_dir + 'valid_set2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        filename = row[1]
        if '.' not in filename: # Skip the header
            continue

        frame = cv2.cvtColor(cv2.imread(os.path.join(args.dataset_dir, filename)), cv2.COLOR_BGR2RGB)

        rvec1, _, _ = predictor1.predict(frame, args.posenet_type == 0)
        rvec2, _, _ = predictor2.predict(frame, args.posenet_type == 0)
        rvec3, _, _ = predictor3.predict(frame, args.posenet_type == 0)
        rvec4, _, _ = predictor4.predict(frame, args.posenet_type == 0)

        rvec = np.average([rvec1, rvec2, rvec3, rvec4], axis=0)

        actual_rvec = np.float32(list(map(float, row[2:5])))

        A, _ = cv2.Rodrigues(rvec)
        B, _ = cv2.Rodrigues(actual_rvec)

        theta = np.arccos((np.trace(A.T @ B) - 1)/ 2)
        theta = np.rad2deg(theta)
        theta = min(theta, 180 - theta)

        print(filename, theta)

        thetas.append(theta)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        A_pose = list(utils.rotation_matrix_to_euler_angles(A))
        A_pose[1] = -A_pose[1]
        B_pose = list(utils.rotation_matrix_to_euler_angles(B))
        B_pose[1] = -B_pose[1]
        utils.draw_axis_rads(frame, *np.radians(A_pose))
        utils.draw_axis_rads(frame, *np.radians(B_pose), cx=(255,255,0), cy=(255,0,255), cz=(0, 255, 255))
        cv2.imwrite(os.path.join(args.output_dir, filename), frame)

print(np.max(thetas))
print(np.min(thetas))
print(np.mean(thetas))