#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

import filename_loaders
import kitti_vo
import plot


SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '11']

def calc_poses(predictions, stamps, stack_size):

    poses = []

    curr_pos = [0, 0]
    positions = [[0, 0]]
    yaw_global = np.deg2rad(0.0)

    for i, prediction in enumerate(predictions[:-stack_size]):

        duration = stamps[i+1] - stamps[i]
        duration_total = stamps[i+stack_size-1] - stamps[i]

        vel_y, vel_x, vel_yaw = prediction / duration_total

        trans_local = np.array([vel_y * duration, vel_x * duration])
        yaw_local = vel_yaw * duration

        rot = np.array([
            [np.sin(yaw_global),  np.cos(yaw_global)],
            [np.cos(yaw_global), -np.sin(yaw_global)]
        ])

        trans_global = rot.dot(trans_local)
        curr_pos += trans_global

        yaw_global = (yaw_global + yaw_local) % (2 * np.pi)

        pose = np.array([
            [ np.cos(yaw_global), 0, np.sin(yaw_global), curr_pos[0]],
            [                  0, 0,                  0,           0],
            [-np.sin(yaw_global), 0, np.cos(yaw_global), curr_pos[1]]
        ], dtype=np.float32)

        poses.append(pose)

    return poses


def write_poses(output_file, poses):
    with open(output_file, 'w') as fd:
        for pose in poses:
            pose_line = ' '.join(map(str, pose.flatten())) + '\n'
            fd.write(pose_line)


def main(model_file, stack_size, input_dir, output_dir):

    model = load_model(model_file, custom_objects={'weighted_mse': kitti_vo.weighted_mse})

    image_paths, stamps, odom, num_outputs = kitti_vo.load_filenames(input_dir, 'odom', stack_size, sequences=SEQUENCES)

    for sequence, (image_paths_, stamps_, odom_) in zip(SEQUENCES, zip(image_paths, stamps, odom)):

        print('Sequence: {}'.format(sequence))

        image_paths, stamps, odom = kitti_vo.stack_data([image_paths_], [stamps_], [odom_], stack_size, test_phase=True)
        image_stacks = kitti_vo.load_image_stacks(image_paths)

        predictions = model.predict(image_stacks)
        predictions *= kitti_vo.ODOM_SCALES

        poses = calc_poses(predictions, stamps_, stack_size)

        output_file = os.path.join(output_dir, '{}.txt'.format(sequence))
        write_poses(output_file, poses)

        # vels = filename_loaders.poses_to_velocities(stamps_, poses, stack_size)
        # plot.plot_trajectory_2d(predictions, vels, stamps, stack_size)
        # plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('stack_size', type=int, help='Stack size')
    parser.add_argument('input_dir', help='Base directory')
    parser.add_argument('output_dir', help='Results directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.model_file, args.stack_size, args.input_dir, args.output_dir)
