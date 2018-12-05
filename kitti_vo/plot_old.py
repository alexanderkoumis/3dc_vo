#!/usr/bin/env python3

import os
from copy import deepcopy
from dateutil.parser import parser

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np
from keras.models import Sequential, load_model

import main


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
plotter_b = Plotter(3, 1)


class ImageLoader(object):
    def __init__(self):
        self.cache = {}
    def load_image(self, path):
        if path not in self.cache:
            self.cache[path] = cv2.imread(path)
        return self.cache[path]

loader = ImageLoader()


def plot_velocities_2d(predictions, ground_truth):

    range_pred = range(len(predictions))
    range_gt = range(len(ground_truth))

    for i, label in enumerate(['y', 'x', 'theta']):

        ax = plotter_b.add_subplot()
        ax.set_ylabel(label)

        line_pred = Line2D(range_pred, predictions[:, i], color='r')
        line_gt = Line2D(range_gt, ground_truth[:, i], color='b')

        ax.add_line(line_pred)
        ax.add_line(line_gt)

        ax.set_xlim(0, len(predictions))
        ax.set_ylim(min(predictions[:,i].min(), ground_truth[:,i].min()),
                    max(predictions[:,i].max(), ground_truth[:,i].max()))


def get_trajectory_2d(odoms, stamps):

    curr_pos = [0, 0]
    positions = [[0, 0]]
    theta_global = 0.0

    for i in range(min(len(stamps)-1, len(odoms)-1)):

        if i + 5 - 1 >= len(stamps):
            print('Brownie fudge')
            break

        duration = stamps[i+1] - stamps[i]
        duration_total = stamps[i+5-1] - stamps[i]

        vel_y, vel_x, vel_theta = odoms[i]

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


def plot_trajectory_2d(predictions, ground_truth, stamps):

    positions = get_trajectory_2d(predictions, stamps)
    positions_gt = get_trajectory_2d(ground_truth, stamps)

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



def plot_latlon(image_dir, odom_dir):
    image_paths = os.listdir(image_dir)
    image_names = [path.split('.')[0] for path in image_paths]
    image_names.sort(key=int)
    x, y = [], []
    for image_name in image_names:
        full_path = os.path.join(odom_dir, '{}.txt'.format(image_name))
        with open(full_path, 'r') as fd:
            data = fd.read().split()
            lat, lon = float(data[0]), float(data[1])    
            x.append(lon)
            y.append(lat)
    line = Line2D(x, y)
    ax = plotter_a.add_subplot()
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))


seq_num = 8
stack_size = 5

dataset_type = 'odom'

kitti_dir = '/home/ubuntu/Development/kitti_vo'
# kitti_dir = '/Users/alexander/Development/kitti_vo'
base_dir = os.path.join(kitti_dir, 'datasets', 'odom', '160_90')
sequences_dir = os.path.join(base_dir, 'sequences')
model_file = os.path.join(kitti_dir, 'models', 'model_odom.h5')
results_dir = os.path.join(kitti_dir, 'results')

seq_name = os.listdir(sequences_dir)[seq_num]
print(seq_name)


image_dir = os.path.join(base_dir, seq_name, 'image_02', 'data')
odom_dir = os.path.join(base_dir, seq_name, 'oxts', 'data')

image_paths, stamps, odom, num_outputs = main.load_filenames(base_dir, dataset_type, stack_size)
image_paths, stamps, odom = image_paths[seq_num], stamps[seq_num], odom[seq_num]
image_paths, stamps, odom = main.stack_data([image_paths], [stamps], [odom], stack_size, test_phase=True)

image_stacks = main.load_image_stacks(image_paths)

odom_gt = np.array(odom)
odom_gt *= main.ODOM_SCALES

seperate = False
multiple_pics = False


model = load_model(model_file, custom_objects={'weighted_mse': main.weighted_mse})
predictions = model.predict(image_stacks)
predictions *= main.ODOM_SCALES


# predictions[:, 2] = savgol_filter(predictions[:, 2], 25, 3)

plot_trajectory_2d(predictions, odom_gt, stamps)
# plot_latlon(image_dir, odom_dir)
plot_velocities_2d(predictions, odom_gt)

plt.show()