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

    line_x_pred = Line2D(range_pred, predictions[:, 0], color='b')
    line_x_gt = Line2D(range_gt, ground_truth[:, 0], color='r')
    line_y_pred = Line2D(range_pred, predictions[:, 1], color='b')
    line_y_gt = Line2D(range_gt, ground_truth[:, 1], color='r')

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


def get_trajectory_2d(velocities, timestamps):

    # curr_pos = np.array([0.0, 0.0, 0.0])
    curr_pos = [0, 0]
    positions = [[0, 0]]

    for i in range(len(velocities)-1):

        stamp_curr = timestamps[i]
        stamp_next = timestamps[i+1]

        vel = velocities[i]
        vel_x, vel_y = vel[0], vel[1]

        elapsed = (stamp_next - stamp_curr) / 1000000.0

        curr_pos[0] += vel_x * elapsed
        curr_pos[1] += vel_y * elapsed
        positions.append(deepcopy(curr_pos))

    return positions


def plot_trajectory_2d(predictions, ground_truth, timestamps):

    positions = get_trajectory_2d(predictions, timestamps)
    positions_gt = get_trajectory_2d(ground_truth, timestamps)

    ax = plotter_a.add_subplot()

    for poss in [positions, positions_gt]:
        x = [pos[0] for pos in poss]
        y = [pos[1] for pos in poss]
        line = Line2D(x, y)
        ax.add_line(line)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))


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


seq_num = 0
stack_size = 3

dataset_type = 'raw'
base_dir = '/home/ubuntu/Development/kitti_vo/datasets/raw/160_90'
model_file = '/home/ubuntu/Development/kitti_vo/raw/model.h5'

seq_name = os.listdir(base_dir)[seq_num]

image_dir = os.path.join(base_dir, seq_name, 'image_02', 'data')
odom_dir = os.path.join(base_dir, seq_name, 'oxts', 'data')

image_paths, stamps, odom, num_outputs = main.load_filenames(base_dir, dataset_type)
image_paths, stamps, odom = image_paths[seq_num], stamps[seq_num], odom[seq_num]
image_paths, stamps_stacks, odom_stacks = main.stack_data([image_paths], [stamps], [odom], stack_size)

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
