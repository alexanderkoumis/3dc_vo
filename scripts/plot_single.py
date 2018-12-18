#!/usr/bin/env python3

import os
import sys
from copy import deepcopy
from dateutil.parser import parser


import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np
from keras.models import load_model

import train


class Plotter(object):
    def __init__(self, rows=1, cols=2):
        self.num = 0
        self.rows = rows
        self.cols = cols
        self.figure = plt.figure()
    def add_subplot(self):
        self.num += 1
        return self.figure.add_subplot(self.rows, self.cols, self.num)

plotter_a = Plotter(1, 1)
plotter_b = Plotter(2, 1)


class ImageLoader(object):
    def __init__(self):
        self.cache = {}
    def load_image(self, path):
        if path not in self.cache:
            self.cache[path] = cv2.imread(path)
        return self.cache[path]

loader = ImageLoader()



def averaged_prediction(predictions, stack_size, idx):
    pred = np.array([0.0, 0.0])
    for j in range(stack_size):
        for k in range(j, stack_size):
            pred += predictions[idx+j] / (stack_size * (k+1))
    return pred



def plot_velocities_2d(predictions, ground_truth):

    range_pred = range(len(predictions))
    range_gt = range(len(ground_truth))

    for i, label in enumerate(['y', 'yaw']):

        ax = plotter_b.add_subplot()
        ax.set_ylabel(label)

        line_pred = Line2D(range_pred, predictions[:, i], color='r')
        line_gt = Line2D(range_gt, ground_truth[:, i], color='b')

        ax.add_line(line_pred)
        ax.add_line(line_gt)

        ax.set_xlim(0, len(predictions))
        ax.set_ylim(min(predictions[:,i].min(), ground_truth[:,i].min()),
                    max(predictions[:,i].max(), ground_truth[:,i].max()))


def get_trajectory_2d(odoms, stamps, stack_size):

    curr_pos = [0, 0]
    positions = [[0, 0]]
    theta_global = 0.0

    for i in range(min(len(stamps)-1, len(odoms)-1)):

        if i + stack_size - 1 >= len(stamps):
            break

        if odoms[i].shape[0] == 1:
            break

        duration = stamps[i+1] - stamps[i]
        duration_total = stamps[i+stack_size-1] - stamps[i]

        prediction = odoms[i]
        # prediction = averaged_prediction(odoms, stack_size, i)

        # offset_y, offset_x, offset_theta = odoms[i]
        # vel_y, vel_x, vel_theta = odoms[i] / duration_total
        vel_y, vel_theta = prediction / duration_total
        vel_x = 0.0

        trans_local = np.array([vel_x * duration, vel_y * duration])
        theta_local = vel_theta * duration

        rot = np.array([
            [np.sin(theta_global),  np.cos(theta_global)],
            [np.cos(theta_global), -np.sin(theta_global)]
        ])

        trans_global = rot.dot(trans_local)

        curr_pos += trans_global

        theta_global = (theta_global + theta_local) % (2 * np.pi)

        positions.append(deepcopy(curr_pos))

    return positions

def plot_trajectory_2d(predictions, ground_truth, stamps, stack_size):

    positions = get_trajectory_2d(predictions, stamps, stack_size)
    positions_gt = get_trajectory_2d(ground_truth, stamps, stack_size)

    ax = plotter_a.add_subplot()
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    min_min, max_max = np.inf, -np.inf

    for poss, color in zip([positions, positions_gt], ['g', 'b']):
        x = [pos[0] for pos in poss]
        y = [pos[1] for pos in poss]

        line = Line2D(x, y, color=color)
        ax.add_line(line)
        min_x = min(min_x, min(x))
        max_x = max(max_x, max(x))
        min_y = min(min_y, min(y))
        max_y = max(max_y, max(y))
        max_max = max(max_x, max_y)
        min_min = min(min_x, min_y)

    ax.set_xlim(min_min, max_max)
    ax.set_ylim(min_min, max_max)



stack_size = 5



kitti_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

dataset_dir = os.path.join(kitti_dir, 'datasets', 'odom', '160_90')
sequences_dir = os.path.join(dataset_dir, 'sequences')

model_file_yaw = os.path.join(kitti_dir, 'models', 'yaw', 'model_odom.0144-0.001839-0.001990.h5')
model_file_y = os.path.join(kitti_dir, 'models', 'y', 'model_odom.0065-2.659613-5.205640.h5')

results_dir = os.path.join(kitti_dir, 'results')


seq_name = '00' if len(sys.argv) == 1 else sys.argv[1]
print(seq_name)

image_paths, stamps, odom, num_outputs = train.load_filenames(dataset_dir, 'odom', stack_size, sequences=[seq_name])
image_paths, stamps, odom = train.stack_data(image_paths, stamps, odom, stack_size, test_phase=True)
image_stacks = train.load_image_stacks(image_paths)


odom_gt = np.array(odom)
odom_gt *= train.ODOM_SCALES


image_stacks = [
    np.expand_dims(image_stacks[:, :, :, :, 0], axis=4),
    np.expand_dims(image_stacks[:, :, :, :, 1], axis=4),
    np.expand_dims(image_stacks[:, :, :, :, 2], axis=4)
]


def weighted_mse(y_true, y_pred):
    import keras.backend as K
    HIGH_ANGLE = 0.1
    mask_gt = K.cast(K.abs(y_true) > HIGH_ANGLE, np.float32) * np.array([3.0])
    mask_lt = K.cast(K.abs(y_true) < HIGH_ANGLE, np.float32) * np.array([1.0])
    return K.mean(K.square(y_true - y_pred) * mask_gt + K.square(y_true - y_pred) * mask_lt)



# model_y = load_model(model_file_y)
# predictions_y = model_y.predict(image_stacks)
# predictions_y *= train.ODOM_SCALES
predictions_y = odom_gt[:, 0].reshape(-1, 1)

model_yaw = load_model(model_file_yaw, custom_objects={'weighted_mse': weighted_mse})
predictions_yaw = model_yaw.predict(image_stacks)
predictions_yaw *= train.ODOM_SCALES

# predictions_yaw = model_yaw.predict(yaw_input).mean(axis=(1, 2, 3, 4))
# predictions_yaw *= train.ODOM_SCALES
# predictions_yaw = predictions_yaw.reshape(-1, 1)

# for i in range(len(predictions_yaw)):
#     if abs(predictions_yaw[i]) < 0.03:
#         predictions_yaw[i] *= 0.2

# predictions_yaw[:, 0] = savgol_filter(predictions_yaw[:, 0], 31, 3)

predictions = np.hstack((predictions_y, predictions_yaw))

plot_trajectory_2d(predictions, odom_gt, stamps, stack_size)
plot_velocities_2d(predictions, odom_gt)

plt.show()
