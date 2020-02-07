import os
import cv2
import csv
import numpy as np
from posenet import PoseNet
import utils

predictor = PoseNet.Predictor('models\\posenet101.pkl', 0)

def load(path):
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    
    head = rows.index('{') + 1
    tail = rows.index('}')

    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    points = np.array([tuple([float(point) for point in coords]) for coords in coords_set])
    return points

thetas = []

with open('Data\\RAW\\valid_set\\validation_set.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        filename = row[0]
        if '.' not in filename:
            continue

        frame = cv2.cvtColor(cv2.imread('Data\\RAW\\valid_set\\images\\' + filename), cv2.COLOR_BGR2RGB)

        pt2d = load('Data\\RAW\\valid_set\\images\\' + filename.replace('.png', '.pts'))
        x_min = min(pt2d[:,1])
        y_min = min(pt2d[:,0])
        x_max = max(pt2d[:,1])
        y_max = max(pt2d[:,0])

        frame = frame[int(x_min):int(x_max), int(y_min):int(y_max), :]
        rvec, pose, img = predictor.predict(frame, True)

        actual_rvec = np.float32(list(map(float, row[1:4])))

        A, _ = cv2.Rodrigues(rvec)
        B, _ = cv2.Rodrigues(actual_rvec)

        theta = np.arccos((np.trace(A.T @ B) - 1 )/ 2)
        theta = np.rad2deg(theta)

        thetas.append(min(abs(theta), abs(180 - theta)))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        utils.draw_axis_rads(frame, *np.radians(list(utils.rotation_matrix_to_euler_angles(A))))
        utils.draw_axis_rads(frame, *np.radians(list(utils.rotation_matrix_to_euler_angles(B))), cx=(255,255,0), cy=(255,0,255), cz=(0, 255, 255))
        cv2.imwrite(os.path.join('output\\images\\validation', filename), frame)

        
print(np.mean(thetas))