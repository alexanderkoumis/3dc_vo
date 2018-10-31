#!/usr/bin/env python3

import argparse
from math import atan2, cos, sin, degrees

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.lines import Line2D

from main import load_filenames, DEFAULT_STACK_SIZE, stack_data, calc_velocity, stack_data


class Plotter(object):
    def __init__(self):
        self.num = 0
        self.figure = plt.figure()
    def add_subplot(self):
        self.num += 1
        return self.figure.add_subplot(2, 2, self.num)
plotter = Plotter()


def plot_line(x, y):
    line = Line2D(x, y)
    ax = plotter.add_subplot()
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    return ax


def plot_ground_truth(poses):
    x_all = [pose[0, 3] for pose in poses]
    y_all = [pose[1, 3] for pose in poses]
    ax = plot_line(x_all, y_all)
    counter = 0
    interval = 10
    for x, y, pose in zip(x_all, y_all, poses):
        x, y = pose[0, 3], pose[1, 3]
        pose = pose[:3, :3].T
        theta = degrees(atan2(pose[2, 1], pose[1, 1]))
        rot = np.array([
            [ sin(theta), cos(theta)],
            [-cos(theta), sin(theta)]
            # [ cos(theta), -sin(theta)],
            # [ sin(theta),  cos(theta)]
        ])
        line = np.matmul(rot, np.array([0.4, 0.0]))
        line = Line2D([x, x+line[0]], [y, y+line[1]], color='r')
        counter += 1
        if counter == interval:
            ax.add_line(line)
            counter = 0




def plot_translations(translations):

    x_all = []
    y_all = []
    curr_rot = np.eye(2)
    curr_pos = np.array([0.0, 0.0])

    for translation in translations:

        x, y = translation

        x_all.append(curr_pos[0])
        y_all.append(curr_pos[1])

        curr_pos += translation

        # curr_pos += curr_rot.dot(translation)

        # angle = atan2(y, x)

        # new_rot = np.array([
        #     [ sin(angle), cos(angle)],
        #     [-cos(angle), sin(angle)]
        #     # [ cos(angle), -sin(angle)],
        #     # [ sin(angle),  cos(angle)]
        # ])

        # curr_rot = np.matmul(new_rot, curr_rot)

    plot_line(x_all, y_all)



def calc_movement(model, image_paths, stamps, poses):

    p_vel = []
    g_vel = []
    p_trans = []
    g_trans = []

    for image_path_stack, stamp_stack, pose_stack in zip(image_paths, stamps, poses):

        time_elapsed = stamp_stack[1] - stamp_stack[0]

        images = [cv2.imread(path) / 255.0 for path in image_path_stack]
        images = np.dstack(images)[np.newaxis]

        pred_velocity = model.predict(images).ravel()
        ground_velocity = calc_velocity(stamp_stack, pose_stack)

        pred_translation = pred_velocity * time_elapsed
        ground_translation = ground_velocity * time_elapsed

        p_vel.append(pred_velocity)
        g_vel.append(ground_velocity)
        p_trans.append(pred_translation)
        g_trans.append(ground_translation)

    return p_trans, g_trans, p_vel, g_vel


def plot_velocities(pred_velocities, ground_velocities):

    nums = list(range(len(pred_velocities)))

    pred_x = [p[0] for p in pred_velocities]
    pred_y = [p[1] for p in pred_velocities]
    ground_x = [p[0] for p in ground_velocities]
    ground_y = [p[1] for p in ground_velocities]

    ax = plotter.add_subplot()

    ax.add_line(Line2D(nums, pred_x, color='r'))
    ax.add_line(Line2D(nums, pred_y, color='g'))
    ax.add_line(Line2D(nums, ground_x, color='b'))
    ax.add_line(Line2D(nums, ground_y, color='y'))
    ax.set_xlim(0, len(nums))
    ax.set_ylim(min(min(pred_y), min(ground_y)), max(max(pred_y), max(ground_y)))


def main(args):

    model = load_model(args.model_file)

    image_paths, stamps, poses = load_filenames(args.base_dir)

    image_paths, stamps, poses = image_paths[args.seq_num], stamps[args.seq_num], poses[args.seq_num]

    image_path_stacks, stamp_stacks, pose_stacks = stack_data([image_paths], [stamps], [poses], 3)

    p_trans, g_trans, p_vel, g_vel = calc_movement(model, image_path_stacks, stamp_stacks, pose_stacks)

    plot_velocities(p_vel, g_vel)
    plot_ground_truth(poses)
    plot_translations(p_trans)
    plot_translations(g_trans)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Base directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('seq_num', type=int, help='KITTI sequence number')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
