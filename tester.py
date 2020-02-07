import os
import cv2
import csv
import numpy as np
from posenet import PoseNet, Predictor
import utils
import glob
import argparse

parser = argparse.ArgumentParser(description='Head pose estimation using the Posenet network.')
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
parser.add_argument('--posenet_type', dest='posenet_type', help='The type of PoseNet to test 0 - Regular, 1 - PnP',
        default=0, type=int)
parser.add_argument('--output_dir', dest='output_dir', help='The output directory',
        default='output\\posenet\\', type=str)
parser.add_argument('--dataset_dir', dest='dataset_dir', help='The test set directory',
        default='Data\\RAW\\test_set\\', type=str)
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

print('test started')

with open(os.path.join(args.output_dir + 'output.csv'), 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['file name', 'rx', 'ry', 'rz'])

    for filename in glob.glob(args.dataset_dir + '*.png'):
        name = filename.replace(args.dataset_dir, '')
        frame = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

        rvec1, _, _ = predictor1.predict(frame, args.posenet_type == 0)
        rvec2, _, _ = predictor2.predict(frame, args.posenet_type == 0)
        rvec3, _, _ = predictor3.predict(frame, args.posenet_type == 0)
        rvec4, _, _ = predictor4.predict(frame, args.posenet_type == 0)

        rvec = np.average([rvec1, rvec2, rvec3, rvec4], axis=0)

        spamwriter.writerow([name, *rvec])

        A, _ = cv2.Rodrigues(rvec)
        pose = list(utils.rotation_matrix_to_euler_angles(A))

        print(name, rvec)

        pose[1] = -pose[1] # flip the pitch for drawing
        utils.draw_axis_rads(frame, *np.radians(pose))

        cv2.imwrite(args.output_dir + name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))