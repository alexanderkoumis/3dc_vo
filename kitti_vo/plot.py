#!/usr/bin/env python3

import os
from copy import deepcopy
from dateutil.parser import parser

import matplotlib.pyplot as plt
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

plotter_a = Plotter()
plotter_b = Plotter(2, 1)


def plot_velocities_2d(predictions, ground_truth):

    ax_x = plotter_b.add_subplot()
    ax_y = plotter_b.add_subplot()

    range_pred = range(len(predictions))
    range_gt = range(len(ground_truth))

    line_x_pred = Line2D(range_pred, predictions[:, 0], color='r')
    line_x_gt = Line2D(range_gt, ground_truth[:, 0], color='b')
    line_y_pred = Line2D(range_pred, predictions[:, 1], color='r')
    line_y_gt = Line2D(range_gt, ground_truth[:, 1], color='b')

    ax_x.add_line(line_x_pred)
    ax_x.add_line(line_x_gt)
    ax_y.add_line(line_y_pred)
    ax_y.add_line(line_y_gt)

    ax_x.set_xlim(0, len(predictions))
    ax_y.set_xlim(0, len(predictions))
    ax_x.set_ylim(min(predictions[:,0].min(), ground_truth[:,0].min()),
                  max(predictions[:,0].max(), ground_truth[:,0].max()))
    ax_y.set_ylim(min(predictions[:,1].min(), ground_truth[:,1].min()),
                  max(predictions[:,1].max(), ground_truth[:,1].max()))


def get_trajectory_2d(odoms, stamps):

    curr_pos = [0, 0]
    positions = [[0, 0]]
    theta_global = 0.0

    for i in range(min(len(stamps)-1, len(odoms)-1)):

        duration = stamps[i+1] - stamps[i]
        odom = odoms[i]
        vel_x, vel_y, vel_theta = odom[0], odom[1], odom[2]

        trans_local = np.array([vel_x * duration, vel_y * duration])

        rot = np.array([
            [np.sin(theta_global),  np.cos(theta_global)],
            [np.cos(theta_global), -np.sin(theta_global)]
        ])

        trans_global = rot.dot(trans_local)

        curr_pos += trans_global

        theta_local = vel_theta * duration
        theta_global = (theta_global + theta_local) % (2 * np.pi)

        positions.append(deepcopy(curr_pos))

    return positions


def plot_trajectory_2d(predictions, ground_truth, stamps):

    positions = get_trajectory_2d(predictions, stamps)
    positions_gt = get_trajectory_2d(ground_truth, stamps) * np.array([-1, 1])

    ax = plotter_a.add_subplot()
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf

    for poss, color in zip([positions, positions_gt], ['g', 'b']):
        x = [pos[0] for pos in poss]
        y = [pos[1] for pos in poss]

        line = Line2D(x, y, color=color)
        ax.add_line(line)
        min_x = min(min_x, min(x))
        max_x = max(max_x, max(x))
        min_y = min(min_y, min(y))
        max_y = max(max_y, max(y))

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


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


seq_num = 6
stack_size = 5

dataset_type = 'raw'
base_dir = '/home/ubuntu/Development/kitti_vo/datasets/raw/160_90'
model_file = '/home/ubuntu/Development/kitti_vo/models/model_raw.h5'

seq_name = os.listdir(base_dir)[seq_num]

image_dir = os.path.join(base_dir, seq_name, 'image_02', 'data')
odom_dir = os.path.join(base_dir, seq_name, 'oxts', 'data')

image_paths, stamps, odom, num_outputs = main.load_filenames(base_dir, dataset_type, stack_size)
image_paths, stamps, odom = image_paths[seq_num], stamps[seq_num], odom[seq_num]
image_paths, stamps, _ = main.stack_data([image_paths], [stamps], [odom], stack_size)

model = load_model(model_file)

predictions = []
for image_stack in image_paths:
    images = [cv2.imread(path) / 255.0 for path in image_stack]
    stacked_images = np.dstack(images)[np.newaxis]
    prediction = model.predict(stacked_images).ravel()
    predictions.append(prediction)
predictions = np.array(predictions)
odom = np.array(odom)

plot_trajectory_2d(predictions, odom, stamps)
plot_latlon(image_dir, odom_dir)
plot_velocities_2d(predictions, odom)

plt.show()
