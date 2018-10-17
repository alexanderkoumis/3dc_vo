#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
from keras.models import load_model

import common


def calc_poses(predictions, stamps, stack_size):
    """Given the models' yaw and y (local) predictions, compute the (global) 3x4 transformation
    matrixes by integrating the local predictions

    Args:
        predictions (np.array): A Nx2 array where N == the number of predictions, the first column
                                contains the y estimates, and the second contains the yaw estimates
        stamps (np.array): Timestamps for each prediction
        stack_size (int): Stack size

    Returns:
        list(np.array): A list of global poses making up the entire trajectory
    """
    last_prediction = predictions[-1].reshape(-1, 2)
    predictions = np.vstack((predictions, np.repeat(last_prediction, axis=0, repeats=stack_size-2)))

    starting_pose = np.hstack((np.eye(3), np.zeros((3, 1)))).astype(np.float32)
    poses = [starting_pose]

    curr_pos = [0, 0]
    positions = [[0, 0]]
    yaw_global = np.deg2rad(0.0)

    for i, prediction in enumerate(predictions):

        try:
            duration = stamps[i+1] - stamps[i]
            duration_total = stamps[i+stack_size-1] - stamps[i]
        except:
            pass

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


def write_poses(output_dir, sequence, poses):
    """Write the poses to a file in the format needed by the official KITTI evaluation tool

    Args:
        output_dir (str): Directory to write the poses to
        sequence (str): The current sequence number
        poses (list(np.array)): The list of global poses
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(args.output_dir, '{}.txt'.format(sequence))

    with open(output_file, 'w') as fd:
        for pose in poses:
            pose_line = ' '.join(map(str, pose.flatten())) + '\n'
            fd.write(pose_line)


def main(args):
    """Evaluate all image sequences using the specified models"""

    # If the reproduction flag is set, scale the yaw predictions by this constant amount
    # because during training of the models used in the paper, this was the amount the
    # outputs were scaled by.
    scale_yaw = 0.37534439 if args.reproduce else 1.0

    sequences = {
        'val': common.sequences_val,
        'test': common.sequences_test
    }[args.seq]

    model_y = load_model(args.model_y)
    model_yaw = load_model(args.model_yaw, custom_objects={'weighted_mse': common.weighted_mse})

    image_paths, stamps, odoms = common.load_filenames_odom(args.input_dir, sequences)

    for sequence, (image_paths_, stamps_, odoms_) in zip(sequences, zip(image_paths, stamps, odoms)):

        print('Sequence: {}'.format(sequence))

        image_paths, stamps, odom = common.stack_data([image_paths_], [stamps_], [odoms_], common.stack_size)

        image_stacks = common.load_image_stacks(image_paths, args.reproduce)
        image_stacks = common.split_image_channels(image_stacks)

        predictions_y = model_y.predict(image_stacks).reshape(-1, 1)
        predictions_yaw = model_yaw.predict(image_stacks).reshape(-1, 1) * scale_yaw

        predictions = np.hstack((predictions_y, predictions_yaw))

        poses = calc_poses(predictions, stamps_, common.stack_size)

        write_poses(args.output_dir, sequence, poses)


def parse_args():

    kitti_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    images_dir = os.path.join(kitti_dir, 'data', 'images')
    output_dir = os.path.join(kitti_dir, 'output')
    model_file_yaw = os.path.join(kitti_dir, 'data', 'models', 'model_pretrained_yaw.h5')
    model_file_y = os.path.join(kitti_dir, 'data', 'models', 'model_pretrained_y.h5')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=images_dir, help='Base directory')
    parser.add_argument('--output_dir', default=output_dir, help='Results directory')
    parser.add_argument('--reproduce', action='store_true', help='Reproduce results from paper')
    parser.add_argument('--model_yaw', default=model_file_yaw, help='Yaw model .h5 file')
    parser.add_argument('--model_y', default=model_file_y, help='Y model .h5 file')
    parser.add_argument('--seq', default='test', choices=['val', 'test'], help='Validation or test seqs.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
