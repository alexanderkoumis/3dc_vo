#!/usr/bin/env python3

import argparse
import random
import sys

import cv2
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU, MaxPooling3D
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from filename_loaders import load_filenames_raw, load_filenames_odom
from image_loader import ImageLoader


# Forward velocity, leftward velocity, angular velocity
# Precalculated to scale output during training and testing
ODOM_IMPORTANCE_SCALES = np.array([0.5, 0.2, 1.0])

# # Velocities
# ODOM_SCALES = np.array([26.4518183, 5.70262262, 1.50662341])

# Offsets
ODOM_SCALES = np.array([10.97064094, 2.36517883, 0.62509463])

TRAIN_SEQUENCES = ['00', '02', '08', '09']
TEST_SEQUENCES = ['01', '03', '04', '05', '06', '10']

def dataset_generator(image_paths_all, odom_all, rgb_scalers, batch_size, memory):

    image_loader = ImageLoader(memory)

    input_target_list = list(zip(image_paths_all, odom_all))
    random.shuffle(input_target_list)

    while True:

        stacked_images_batch = []
        odom_batch = []

        for paths, odom in input_target_list:

            images = [image_loader.load_image(path).astype(np.float32) for path in paths]

            for idx, image in enumerate(images):

                for channel, scaler in enumerate(rgb_scalers):

                    image_channel = image[:, :, channel]
                    image_channel_flat = image_channel.reshape(-1, 1)
                    image_channel_flat = scaler.transform(image_channel_flat)
                    image_channel = image_channel_flat.reshape(image[:, :, channel].shape)
                    images[idx][:, :, channel] = image_channel

            rows, cols, channels = images[0].shape
            stack_size = len(paths)

            stack_shape = (rows, cols, stack_size, channels)
            image_stack = np.zeros(stack_shape, dtype=np.float32)

            for stack_idx, image in enumerate(images):
                image_stack[:, :, stack_idx, 0] = image[:, :, 0]
                image_stack[:, :, stack_idx, 1] = image[:, :, 1]
                image_stack[:, :, stack_idx, 2] = image[:, :, 2]

            stacked_images_batch.append(image_stack)
            odom_batch.append(odom)

            if len(stacked_images_batch) == batch_size:
                yield np.array(stacked_images_batch), np.array(odom_batch)
                stacked_images_batch = []
                odom_batch = []


def calc_yaw_velocity(stamp_start, stamp_end, yaw_start, yaw_end):

    time_elapsed = stamp_end - stamp_start
    yaw_diff = yaw_end - yaw_start

    if yaw_diff > np.pi:
        yaw_diff -= 2.0 * np.pi
    if yaw_diff < -np.pi:
        yaw_diff += 2.0 * np.pi

    yaw_vel = yaw_diff / time_elapsed
    return yaw_vel


def stack_data(image_paths, stamps, odoms, stack_size, test_phase=False):
    """
    In format:
        image_paths: [
            [sequence 0 image paths],
            [sequence 1 image_paths],
            etc
        ]
        stamps: [
            [sequence 0 stamps],
            [sequence 1 stamps],
            etc
        ]
        poses: [
            [sequence 0 poses],
            [sequence 1 poses],
            etc
        ]

    Out format:
        image_paths/stamps/poses: [
            [stack 0],
            [stack 1],
            etc
        ]
    """

    image_paths_stacks = []
    stamps_new = []
    odoms_new = []

    for image_paths_seq, stamps_seq, odom_seq in zip(image_paths, stamps, odoms):

        for i in range(len(image_paths_seq)-stack_size+1):

            image_paths_stack = [image_paths_seq[i+j] for j in range(stack_size)]
            image_paths_stacks.append(image_paths_stack)
            stamps_new.append(stamps_seq[i])
            odoms_new.append(odom_seq[i])

    if test_phase:
        odoms_new /= ODOM_SCALES
        return image_paths_stacks, stamps_new, odoms_new

    # Break this out into seperate function, only for angular velocity
    high_low_ratio = 1.5
    high_angle_thresh = 0.12
    # high_angle_thresh = 0.03
    high_angle_count = sum(abs(odom[2]) > high_angle_thresh for odom in odoms_new)
    low_angle_keep = int(high_angle_count * high_low_ratio)

    image_paths_stacks_new_new = []
    stamps_new_new = []
    odoms_new_new = []
    idxs = []

    for idx, (path_stack, stamp, odom) in enumerate(zip(image_paths_stacks, stamps_new, odoms_new)):
        if abs(odom[2]) > high_angle_thresh:
            image_paths_stacks_new_new.append(path_stack)
            stamps_new_new.append(stamp)
            odoms_new_new.append(odom)
        else:
            idxs.append(idx)

    keep_idxs = np.random.choice(idxs, low_angle_keep, replace=False)
    for keep_idx in keep_idxs:
        image_paths_stacks_new_new.append(image_paths_stacks[keep_idx])
        stamps_new_new.append(stamps_new[keep_idx])
        odoms_new_new.append(odoms_new[keep_idx])

    odoms_new_new = np.array(odoms_new_new)
    odoms_new_new /= ODOM_SCALES

    return image_paths_stacks_new_new, stamps_new_new, odoms_new_new


def get_input_shape(image_paths):
    stack_size = len(image_paths[0])
    rows, cols, channels = cv2.imread(image_paths[0][0]).shape
    return rows, cols, stack_size, channels


def load_filenames(data_dir, dataset_type, stack_size, sequences=None):
    loader = {
        'raw': load_filenames_raw,
        'odom': load_filenames_odom
    }
    return loader[dataset_type](data_dir, stack_size, sequences)


def load_image_stacks(image_path_stacks):
    """Loads image path stacks into memory"""

    num_stacks = len(image_path_stacks)
    rows, cols, stack_size, channels = get_input_shape(image_path_stacks)

    shape = (num_stacks, rows, cols, stack_size, channels)

    image_stacks = np.zeros(shape, dtype=np.float32)

    paths = set([path for path_stack in image_path_stacks for path in path_stack])

    # Load all images into memory, flatten them, and normalize by channel
    images_flat = np.array([cv2.imread(path).flatten() for path in paths], dtype=np.float32)
    for channel in range(channels):
        start, end = channel * rows * cols, (channel+1) * rows * cols
        channel_col = images_flat[:, start:end].reshape(-1, 1)
        channel_col = StandardScaler().fit_transform(channel_col)
        images_flat[:, start:end] = channel_col.reshape((images_flat.shape[0], rows * cols))

    image_cache = {path: img.reshape((rows, cols, channels)) for path, img in zip(paths, images_flat)}

    for idx, path_stack in enumerate(image_path_stacks):
        stack_shape = (rows, cols, stack_size, channels)
        image_stack = np.zeros(stack_shape, dtype=np.float32)
        for stack_idx, path in enumerate(path_stack):
            image = image_cache[path]
            image_stack[:, :, stack_idx, 0] = image[:, :, 0]
            image_stack[:, :, stack_idx, 1] = image[:, :, 1]
            image_stack[:, :, stack_idx, 2] = image[:, :, 2]
        image_stacks[idx] = image_stack

    return image_stacks


def build_model(input_shape, num_outputs):
    model = Sequential()
    model.add(Conv3D(32, 3, strides=3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv3D(32, 3, strides=3, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Conv3D(16, 3, strides=3, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(num_outputs, activation='linear'))
    return model


def weighted_mse(y_true, y_pred):
    return K.mean(ODOM_IMPORTANCE_SCALES*K.square(y_true - y_pred))


def get_rgb_scalers(image_path_stacks):

    num_stacks = len(image_path_stacks)
    rows, cols, stack_size, channels = get_input_shape(image_path_stacks)

    paths = set([path for path_stack in image_path_stacks for path in path_stack])
    scalers = [StandardScaler() for _ in range(channels)]

    # Load all images into memory, flatten them, and normalize by channel
    images = np.array([cv2.imread(path).flatten() for path in paths], dtype=np.float32)
    for channel, scaler in enumerate(scalers):
        start, end = channel * rows * cols, (channel+1) * rows * cols
        channel_col = images_flat[:, start:end].reshape(-1, 1)
        scaler.fit(channel_col)

    return scalers


def load_data(data_dir, dataset_type, stack_size, memory_type):

    images_train, stamps, odom_train, num_outputs = load_filenames(data_dir, dataset_type,
                                                                   stack_size, TRAIN_SEQUENCES)
    images_train, stamps, odom_train = stack_data(images_train, stamps, odom_train, stack_size)

    images_test, stamps, odom_test, num_outputs = load_filenames(data_dir, dataset_type,
                                                                 stack_size, TEST_SEQUENCES)
    images_test, stamps, odom_test = stack_data(images_test, stamps, odom_train, stack_size)

    input_shape = get_input_shape(images_train)

    if memory_type == 'high':
        images_train = load_image_stacks(images_train)
        images_test = load_image_stacks(images_test)

    return images_train, odom_train, images_test, odom_test, input_shape, num_outputs


def main(args):

    images_train, odom_train, images_test, odom_test, input_shape, num_outputs = load_data(
        args.data_dir, args.dataset_type, args.stack_size, args.memory)

    if args.resume:
        model = load_model(args.model_file, custom_objects={'weighted_mse': weighted_mse})
    else:
        model = build_model(input_shape, num_outputs)
        model.compile(loss=weighted_mse, optimizer='adam')

    model.summary()

    if args.memory == 'high':

        model.fit(images_train, odom_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            validation_data=(images_test, odom_test),
            callbacks=[ModelCheckpoint(args.model_file)])
    else:

        scalers = get_rgb_scalers(images_test)
        model.fit_generator(dataset_generator(images_train, odom_train, scalers, args.batch_size, args.memory),
            epochs=args.epochs,
            steps_per_epoch=int(len(images_train)/args.batch_size),
            validation_steps=int(len(images_test)/args.batch_size),
            verbose=1,
            validation_data=dataset_generator(images_test, odom_test, scalers, args.batch_size, args.memory),
            callbacks=[ModelCheckpoint(args.model_file)])

    print('Saving model to {}'.format(args.model_file))
    model.save(args.model_file)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Dataset directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('dataset_type', help='Dataset type (either raw or odom)')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('-s', '--stack_size', type=int, default=5, help='Size of image stack')
    parser.add_argument('-m', '--memory', default='low', help='Memory strategy, one of (low, medium, high)')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume training on model')
    args = parser.parse_args()

    if args.memory not in {'low', 'medium', 'high'}:
        sys.exit('--memory option must be one of low, medium, or high')

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
