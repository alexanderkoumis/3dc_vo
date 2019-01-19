#!/usr/bin/env python3

import argparse
import copy
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.models import load_model

import filename_loaders
import train


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


def expand_image(image):
    rows, cols, _ = image.shape
    image_large = cv2.resize(image, (cols+2, rows+2))
    image_large = np.expand_dims(image_large, axis=2)
    image_cropped = image_large[1:-1, 1:-1]
    return image_cropped

def expand_stacks(image_stacks):
    stack_size = image_stacks[0].shape[3]
    assert stack_size == 5
    for image_stack_channel in image_stacks:
        for image_stack in image_stack_channel:
            for i in range(stack_size):
                image_stack[:, :, i, :] = expand_image(image_stack[:, :, i, :])
    return image_stacks

def predict(model_yaw, image_stacks, mode):

    if mode == 'normal':

        predictions_yaw = model_yaw.predict(image_stacks)

    elif mode == 'flipped':

        image_stacks_flipped = [np.flip(s, axis=2) for s in image_stacks]
        predictions_yaw = -model_yaw.predict(image_stacks_flipped)

    elif mode == 'merged':

        image_stacks_flipped = [np.flip(s, axis=2) for s in image_stacks]
        predictions_yaw = model_yaw.predict(image_stacks)            
        predictions_yaw_flipped = -model_yaw.predict(image_stacks_flipped)
        predictions_yaw = np.mean((predictions_yaw, predictions_yaw_flipped), axis=0)

    elif mode == 'normal_multiple':

        predictions_yaw_all = []

        for i in range(3):
            predictions_yaw = model_yaw.predict(image_stacks)
            predictions_yaw_all.append(predictions_yaw)
            image_stacks = expand_stacks(image_stacks)

        predictions_yaw = np.mean(predictions_yaw_all, axis=0)

    return predictions_yaw


def load_scalers(input_dir, stack_size):
    image_paths, stamps, poses_gt = filename_loaders.load_filenames_odom(input_dir, stack_size, sequences=train.TRAIN_SEQUENCES)
    image_paths, stamps, poses_gt_stacks = train.stack_data(image_paths, stamps, poses_gt, stack_size)
    _, scalers = train.load_image_stacks(image_paths)
    return scalers


def main(args):

    model_yaw = load_model(args.model_file, custom_objects={'weighted_mse': train.weighted_mse})
    # scalers = load_scalers(args.input_dir, args.stack_size)

    image_paths, stamps, poses_gt = filename_loaders.load_filenames_odom(args.input_dir, args.stack_size, sequences=train.TEST_SEQUENCES)

    for sequence, (image_paths_, stamps_, poses_gt_) in zip(train.TEST_SEQUENCES, zip(image_paths, stamps, poses_gt)):

        image_paths, stamps, poses_gt_stacks = train.stack_data([image_paths_], [stamps_], [poses_gt_], args.stack_size)
        odom_gt_y = np.array([filename_loaders.poses_to_offsets(p, ['y']) for p in poses_gt_stacks]).reshape(-1, 1)

        # image_stacks, _ = train.load_image_stacks(image_paths, scalers)
        image_stacks, _ = train.load_image_stacks(image_paths)
        image_stacks = train.convert(image_stacks)

        predictions_yaw = predict(model_yaw, image_stacks, args.mode).reshape(-1, 1)

        predictions = np.hstack((odom_gt_y, predictions_yaw))

        poses = calc_poses(predictions, stamps_, args.stack_size)

        output_file = os.path.join(args.output_dir, '{}.txt'.format(sequence))
        write_poses(output_file, poses)

    K.clear_session()
    del model_yaw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('stack_size', type=int, help='Stack size')
    parser.add_argument('input_dir', help='Base directory')
    parser.add_argument('output_dir', help='Results directory')
    parser.add_argument('mode', help='Predict mode')
    args = parser.parse_args()

    if args.mode not in {'normal', 'flipped', 'merged', 'normal_multiple'}:
        raise Exception('args.mode must be one of [normal, flipped, merged, normal_multiple]')

    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)
