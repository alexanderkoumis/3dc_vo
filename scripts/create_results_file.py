#!/usr/bin/env python3

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

import filename_loaders
import train
# import plot


def averaged_prediction(predictions, stack_size, idx):
    pred = 0.0
    for j in range(stack_size):
        for k in range(j, stack_size):
            pred += predictions[idx+j] / (stack_size * (k+1))
    return pred


def calc_poses(predictions, stamps, stack_size):

    poses = []

    curr_pos = [0, 0]
    positions = [[0, 0]]
    yaw_global = np.deg2rad(0.0)

    for i, prediction in enumerate(predictions[:-stack_size]):

        duration = stamps[i+1] - stamps[i]
        duration_total = stamps[i+stack_size-1] - stamps[i]

        # prediction = averaged_prediction(predictions, stack_size, i)
        # vel_y, vel_x, vel_yaw = prediction / duration_total
        vel_y, vel_yaw = prediction / duration_total
        vel_x = 0.0

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


def load_image_stacks(sequence, image_paths, cache):

    if cache is not None:

        if sequence not in cache:
            image_stacks = train.load_image_stacks(image_paths)
            cache[sequence] = image_stacks

        return cache[sequence]

    image_stacks = train.load_image_stacks(image_paths)
    return image_stacks


def main(args, cache=None):

    if filename_loaders.YAW:
        model_yaw = load_model(args.model_file)
        model_y = load_model('/home/koumis/Development/kitti_vo/models/odom/y/{}/model_odom.h5'.format(args.stack_size))
    else:
        model_yaw = load_model('/home/koumis/Development/kitti_vo/models/odom/yaw/{}/model_odom.h5'.format(args.stack_size))
        model_y = load_model(args.model_file)

    image_paths, stamps, odom, num_outputs = train.load_filenames(args.input_dir, 'odom', args.stack_size, sequences=train.TEST_SEQUENCES)

    for sequence, (image_paths_, stamps_, odom_) in zip(train.TEST_SEQUENCES, zip(image_paths, stamps, odom)):

        # print('Sequence: {}'.format(sequence))

        image_paths, stamps, odom = train.stack_data([image_paths_], [stamps_], [odom_], args.stack_size, test_phase=True)

        image_stacks = load_image_stacks(sequence, image_paths, cache)
        image_stacks = train.convert(image_stacks)

        predictions_y = model_y.predict(image_stacks)
        predictions_y *= train.ODOM_SCALES

        predictions_yaw = model_yaw.predict(image_stacks)
        predictions_yaw *= train.ODOM_SCALES

        predictions = np.hstack((predictions_y, predictions_yaw))

        poses = calc_poses(predictions, stamps_, args.stack_size)

        output_file = os.path.join(args.output_dir, '{}.txt'.format(sequence))
        write_poses(output_file, poses)


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
    main(args)
