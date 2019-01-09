#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
import sys

import cv2
import numpy as np
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU, MaxPooling3D, Input, Average, SpatialDropout3D
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from filename_loaders import load_filenames_odom, YAW
from image_loader import ImageLoader
from recent_model_renamer import RecentModelRenamer



TRAIN_SEQUENCES = ['00', '02', '08', '09']
TEST_SEQUENCES = ['03', '04', '05', '06', '07', '10']


HIGH_ANGLE = 0.05


def stack_data(image_paths, stamps, odoms, stack_size, test_phase=False):
    """
    In format:
        image_paths: [[sequence 0 image paths], [sequence 1 image_paths], etc]
        stamps: [[sequence 0 stamps], [sequence 1 stamps], etc]
        poses: [[sequence 0 poses], [sequence 1 poses], etc]

    Out format:
        image_paths/stamps/poses: [[stack 0], [stack 1], etc]
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

    return image_paths_stacks, stamps_new, np.array(odoms_new)


def get_input_shape(image_paths):
    stack_size = len(image_paths[0])
    rows, cols, channels = cv2.imread(image_paths[0][0]).shape
    return rows, cols, stack_size, channels


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



def build_channel_model(channel_shape, stack_size):
    if YAW:
        regu = 0.005
        input_layer = Input(shape=channel_shape)
        hidden_layer = Conv3D(8, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = SpatialDropout3D(0.1)(hidden_layer)
        hidden_layer = Conv3D(16, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = SpatialDropout3D(0.1)(hidden_layer)
        hidden_layer = Conv3D(32, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Conv3D(4, [1, 1, stack_size], strides=[1, 1, stack_size], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Flatten()(hidden_layer)
        hidden_layer = Dense(64, kernel_regularizer=l2(regu), activation='relu')(hidden_layer)
        hidden_layer = LeakyReLU()(hidden_layer)
        output_layer = Dense(1, activation='linear')(hidden_layer)
    else:
        regu = 0.1
        input_layer = Input(shape=channel_shape)
        hidden_layer = Conv3D(32, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)
        hidden_layer = Conv3D(16, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)
        hidden_layer = Conv3D(8, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)
        hidden_layer = Conv3D(4, 1, strides=[1, 1, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Conv3D(1, [1, 1, stack_size], strides=[1, 1, stack_size], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Flatten()(hidden_layer)
        hidden_layer = Dense(64, kernel_regularizer=l2(regu), activation='relu')(hidden_layer)
        hidden_layer = LeakyReLU()(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)
        output_layer = Dense(1, activation='linear')(hidden_layer)
    return input_layer, output_layer


def build_model(input_shape, stack_size):

    channel_shape = list(copy.deepcopy(input_shape))

    in_, out_ = build_channel_model(channel_shape, stack_size)
    model = Model(inputs=in_, outputs=out_)

    return model


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


def load_data(data_dir, stack_size):

    images_train, stamps, odom_train = load_filenames_odom(data_dir, stack_size, TRAIN_SEQUENCES)
    images_train, stamps, odom_train = stack_data(images_train, stamps, odom_train, stack_size)

    images_test, stamps, odom_test = load_filenames_odom(data_dir, stack_size, TEST_SEQUENCES)
    images_test, stamps, odom_test = stack_data(images_test, stamps, odom_test, stack_size)

    input_shape = get_input_shape(images_train)

    images_train = load_image_stacks(images_train)
    images_test = load_image_stacks(images_test)

    # Augment dataset 0 (skip)

    # images_train_double = []
    # odom_train_double = []

    # for idx, (image_stack, odom) in enumerate(zip(images_train, odom_train)):
    #     if idx + stack_size - 1 < len(images_train):

    #         image_stack_next = images_train[idx+stack_size-1]
    #         odom_next = odom_train[idx+stack_size-1]

    #         odom_new = odom + odom_next

    #         if np.abs(odom_new) > np.max(odom_train) * 0.8:
    #             continue

    #         # image_stack_new = np.concatenate([
    #         #     np.expand_dims(image_stack[:, :, 0, :], axis=2),
    #         #     np.expand_dims(image_stack[:, :, 2, :], axis=2),
    #         #     np.expand_dims(image_stack[:, :, 4, :], axis=2),
    #         #     np.expand_dims(image_stack_next[:, :, 2, :], axis=2),
    #         #     np.expand_dims(image_stack_next[:, :, 4, :], axis=2)
    #         # ], axis=2)

    #         image_stack_new = np.concatenate([
    #             np.expand_dims(image_stack[:, :, 0, :], axis=2),
    #             np.expand_dims(image_stack[:, :, 2, :], axis=2),
    #             np.expand_dims(image_stack_next[:, :, 2, :], axis=2),
    #         ], axis=2)

    #         images_train_double.append(image_stack_new)
    #         odom_train_double.append(odom_new)

    # images_train = np.concatenate((images_train, images_train_double))
    # odom_train = np.concatenate((odom_train, odom_train_double))

    # Augment dataset 1

    # images_train_flip = []
    # odom_train_flip = []

    # for image_stack, odom in zip(images_train, odom_train):
    #     if abs(odom) > 0.05:
    #         image_stack_flipped = np.flip(image_stack, axis=1)
    #         images_train_flip.append(image_stack_flipped)
    #         odom_train_flip.append(odom * -1.0)

    # images_train = np.concatenate((images_train, images_train_flip))
    # odom_train = np.concatenate((odom_train, odom_train_flip))

    # Augment dataset 2

    images_train_flip = np.flip(images_train, axis=2)
    odom_train_flip = [o * -1.0 for o in odom_train]

    num_stacks = images_train.shape[0]
    images_train_new = np.repeat(images_train, 2, axis=0)
    odom_train_new = np.repeat(odom_train, 2, axis=0)

    print(images_train.shape)
    print(images_train_flip.shape)
    print(images_train_new.shape)

    images_train_new[num_stacks:num_stacks+num_stacks] = images_train_flip
    odom_train_new[num_stacks:num_stacks+num_stacks] = odom_train_flip

    images_train = images_train_new
    odom_train = odom_train_new

    return images_train, odom_train, images_test, odom_test, input_shape


def get_model_tpl(model_file_full):
    model_file_full = os.path.abspath(model_file_full)
    model_file = model_file_full.split('/')[-1]
    directory = '/'.join(model_file_full.split('/')[:-1])
    base, ext = model_file.split('.')
    model_tpl = base + '.{epoch:04d}-{loss:.6f}-{val_loss:.6f}.' + ext
    return os.path.join(directory, model_tpl)


def save_history_file(history_file, history_dict):
    with open(history_file, 'w+') as fd:
        history_str = json.dumps(history_dict)
        fd.write(history_str)


def weighted_mse(y_true, y_pred):
    mask_gt = K.cast(K.abs(y_true) > HIGH_ANGLE, np.float32) * np.array([2.0])
    mask_lt = K.cast(K.abs(y_true) < HIGH_ANGLE, np.float32) * np.array([1.0])
    return K.mean(K.square(y_true - y_pred) * mask_gt + K.square(y_true - y_pred) * mask_lt)



def main(args):

    model_file_tpl = get_model_tpl(args.model_file)
    images_train, odom_train, images_test, odom_test, input_shape = load_data(args.data_dir, args.stack_size)

    if args.resume:
        model = load_model(args.model_file, custom_objects={'weighted_mse': weighted_mse})
    else:
        model = build_model(input_shape, args.stack_size)
        # model.compile(loss='mse', optimizer='adam')
        model.compile(loss=weighted_mse, optimizer='adam')

    model.summary()

    callbacks = [ModelCheckpoint(model_file_tpl), RecentModelRenamer(args.model_file)]

    history = model.fit(images_train, odom_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        shuffle=True,
        validation_data=(images_test, odom_test),
        callbacks=callbacks)

    print('Saving model to {}'.format(args.model_file))
    model.save(args.model_file)

    print('Saving history to {}'.format(args.history_file))
    save_history_file(args.history_file, history.history)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Dataset directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('history_file', help='Output history file')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('-s', '--stack_size', type=int, default=5, help='Size of image stack')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume training on model')
    args = parser.parse_args()

    global HIGH_ANGLE
    HIGH_ANGLE = {
        3: 0.05,
        5: 0.1,
        7: 0.15
    }[args.stack_size]

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
