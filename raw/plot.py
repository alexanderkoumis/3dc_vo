#!/usr/bin/env python3

import os
from copy import deepcopy
from dateutil.parser import parser

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from keras.models import Sequential, load_model

import main


image_dir = './data/image_02/data'
odom_dir = './data/oxts/data'
stamp_file = './data/image_02/timestamps.txt'
model_file = './output.h5'





def get_timestamps(stamp_file):
    parse = parser()
    results = []
    with open(stamp_file, 'r') as fd:
        for line in fd.readlines():
            stamp = parse.parse(line)
            results.append(stamp)
    return results

def plot_velocities_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for p, o in zip(predictions, odom_data):
        ax.scatter(p[0], p[1], p[2], c='r', marker='o')
        ax.scatter(o[0], o[1], o[2], c='b', marker='o')
    plt.show()

def plot_velocities_2d():
    plt.plot(predictions[:, 0], predictions[:, 1], 'ro')
    plt.plot(odom_data[:, 0], odom_data[:, 1], 'bo')
    plt.show()

def get_trajectory_2d(predictions, odom_data, odom_scale, odom_mins, timestamps):

    predictions *= odom_scale
    predictions += odom_mins

    odom_data *= odom_scale
    odom_data += odom_mins

    # curr_pos = np.array([0.0, 0.0, 0.0])
    curr_pos = [0, 0, 0]
    positions = [[0, 0, 0]]

    curr_pos_gt = [0, 0, 0]
    positions_gt = [[0, 0, 0]]

    for i in range(len(predictions)-1):

        stamp_curr = timestamps[i]
        stamp_next = timestamps[i+1]

        pred = predictions[i]
        vel_x, vel_y, vel_z = pred[0], pred[1], pred[2]

        pos_gt = odom_data[i]
        vel_x_gt, vel_y_gt, vel_z_gt = pos_gt[0], pos_gt[1], pos_gt[2]

        elapsed = (stamp_next - stamp_curr).microseconds / 1000000.0

        curr_pos[0] += vel_x * elapsed
        curr_pos[1] += vel_y * elapsed
        curr_pos[2] += vel_z * elapsed
        positions.append(deepcopy(curr_pos))

        curr_pos_gt[0] += vel_x_gt * elapsed
        curr_pos_gt[1] += vel_y_gt * elapsed
        curr_pos_gt[2] += vel_z_gt * elapsed
        positions_gt.append(deepcopy(curr_pos_gt))

    return positions, positions_gt
    
def plot_trajectory_2d(predictions, odom_data, odom_scale, odom_mins, timestamps):
    positions, positions_gt = get_trajectory_2d(predictions, odom_data, odom_scale, odom_mins, timestamps)

    ax = plt.figure().add_subplot(111)

    for poss in [positions, positions_gt]:
        x = [pos[0] for pos in poss]
        y = [pos[1] for pos in poss]
        line = Line2D(x, y)
        ax.add_line(line)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(x), max(x))

    # plt.show()
    
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
    ax = plt.figure().add_subplot(111)
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    plt.show()




image_data, odom_data, odom_mins, odom_scale = main.load_data(image_dir, odom_dir, 4)
timestamps = get_timestamps(stamp_file)

model = load_model(model_file)
predictions = model.predict(image_data)


# plot_velocities_3d()
# plot_velocities_2d()
plot_trajectory_2d(predictions, odom_data, odom_scale, odom_mins, timestamps)

plot_latlon(image_dir, odom_dir)


