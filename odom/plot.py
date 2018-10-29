#!/usr/bin/env python3

import argparse
from math import atan2, cos, sin, degrees

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.lines import Line2D

from main_2d_3s import load_filenames, DEFAULT_STACK_SIZE, stack_data, calc_velocity, stack_data


def plot_line(x, y):
    line = Line2D(x, y)
    ax = plt.figure().add_subplot(111)
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))


def plot_ground_truth(poses):
    x = [pose[0, 3] for pose in poses]
    y = [pose[1, 3] for pose in poses]
    plot_line(x, y)


def get_predicted_translations(model, image_paths, stamps, poses):

    result = []

    for image_path_stack, stamp_stack, pose_stack in zip(image_paths, stamps, poses):

        time_elapsed = stamp_stack[1] - stamp_stack[0]

        images = [cv2.imread(path) / 255.0 for path in image_path_stack]
        images = np.dstack(images)[np.newaxis]
        pred_velocity = model.predict(images)


        ground_velocity = calc_velocity(stamp_stack, pose_stack)

        # cv2.imshow('wow', np.vstack([cv2.imread(path) for path in image_path_stack]))
        # cv2.waitKey(0)
        pred_velocity = ground_velocity

        # print(pred_velocity, ground_velocity)
        pred_translation = pred_velocity.ravel() * time_elapsed

        result.append(pred_translation)

    return result


# def plot_pred_translations(pred_translations):

#     x_all = []
#     y_all = []
#     curr_rot = np.eye(2)
#     curr_pos = np.array([0.0, 0.0])

#     for translation in pred_translations:

#         x, y = translation

#         x_all.append(curr_pos[0])
#         y_all.append(curr_pos[1])

#         # curr_pos += translation

#         curr_pos += curr_rot.dot(translation)

#         angle = atan2(y, x)

#         new_rot = np.array([
#             [ sin(angle), cos(angle)],
#             [-cos(angle), sin(angle)]
#             # [ cos(angle), -sin(angle)],
#             # [ sin(angle),  cos(angle)]
#         ])

#         curr_rot = np.matmul(new_rot, curr_rot)

#     plot_line(x_all, y_all)


def get_velocities(model, image_paths, stamps, poses):

    pred_result = []
    ground_result = []

    for image_path_stack, stamp_stack, pose_stack in zip(image_paths, stamps, poses):

        time_elapsed = stamp_stack[1] - stamp_stack[0]

        images = [cv2.imread(path) / 255.0 for path in image_path_stack]
        images = np.dstack(images)[np.newaxis]
        pred_velocity = model.predict(images).ravel()
        ground_velocity = calc_velocity(stamp_stack, pose_stack)
        pred_result.append(pred_velocity)
        ground_result.append(ground_velocity)

    return pred_result, ground_result



def plot_velocities(pred_velocities, ground_velocities):

    nums = list(range(len(pred_velocities)))


    
    pred_x = [p[0] for p in pred_velocities]
    pred_y = [p[1] for p in pred_velocities]
    ground_x = [p[0] for p in ground_velocities]
    ground_y = [p[1] for p in ground_velocities]

    
    ax = plt.figure().add_subplot(111)

    ax.add_line(Line2D(nums, pred_x, color='r'))
    ax.add_line(Line2D(nums, pred_y, color='g'))
    ax.add_line(Line2D(nums, ground_x, color='b'))
    ax.add_line(Line2D(nums, ground_y, color='y'))
    ax.set_xlim(0, len(nums))

    # ax.set_xlim(min(x), max(x))
    # ax.set_ylim(min(y), max(y))
    


def main(args):

    model = load_model(args.model_file)

    image_paths, stamps, poses = load_filenames(args.base_dir)

    image_paths, stamps, poses = image_paths[args.seq_num], stamps[args.seq_num], poses[args.seq_num]

    image_path_stacks, stamp_stacks, pose_stacks = stack_data([image_paths], [stamps], [poses], 3)

    # pred_translations = get_predicted_translations(model, image_path_stacks, stamp_stacks, pose_stacks)

    pred_velocities, ground_velocities = get_velocities(model, image_path_stacks, stamp_stacks, pose_stacks)

    plot_velocities(pred_velocities, ground_velocities)

    # plot_ground_truth(poses)
    # plot_pred_translations(pred_translations)


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
