#!/usr/bin/env python3

import argparse
import copy
import json
import os
import pickle
import random
import sys

import cv2
import numpy as np
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D, Conv3D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU, MaxPooling3D, Input, Average
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from filename_loaders import load_filenames_odom, YAW, poses_to_offsets
from image_loader import ImageLoader
from recent_model_renamer import RecentModelRenamer



TRAIN_SEQUENCES = ['00', '02', '08', '09']
TEST_SEQUENCES = ['03', '04', '05', '06', '07', '10']
# TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '07', '08', '09']
# TEST_SEQUENCES = ['05', '06', '10']



HIGH_ANGLE = 0.05
ODOM_SCALES = np.array([0.37534439])

image_loader = ImageLoader()


def dataset_generator(image_paths_all, odoms_all, rgb_scalers, batch_size):

    global image_loader
    input_target_list = list(zip(image_paths_all, odoms_all))
    # random.shuffle(input_target_list)

    while True:

        stacked_images_batch = [[], [], []]
        odom_batch = []

        for paths, odoms in input_target_list:

            try:
                images = [image_loader.load_image(path).astype(np.float32) for path in paths]
            except:
                continue

            rows, cols = images[0].shape[:2]
            stack_size = len(paths)
            stack_shape = (rows, cols, stack_size, 1)
            image_stacks = [np.zeros(stack_shape, dtype=np.float32) for _ in range(3)]

            for idx, image in enumerate(images):

                for channel, scaler in enumerate(rgb_scalers):

                    image_channel = image[:, :, channel]
                    image_channel_flat = image_channel.reshape(-1, 1)
                    image_channel_flat = scaler.transform(image_channel_flat)
                    image_channel = image_channel_flat.reshape(image[:, :, channel].shape)
                    image_stacks[channel][:, :, idx] = image_channel[:, :, np.newaxis]

            for channel in range(3):
                stacked_images_batch[channel].append(image_stacks[channel])

            odom_batch.append(odoms)

            if len(stacked_images_batch[0]) == batch_size:
                yield [np.array(batch) for batch in stacked_images_batch], np.array(odom_batch)
                stacked_images_batch = [[], [], []]
                odom_batch = []


def stack_data(image_paths, stamps, poses, stack_size, augment=False):
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
    poses_new = []

    for image_paths_seq, stamps_seq, pose_seq in zip(image_paths, stamps, poses):

        for i in range(len(image_paths_seq)-stack_size+1):

            paths_stack = [image_paths_seq[i+j] for j in range(stack_size)]
            stamp_stack = [stamps_seq[i+j] for j in range(stack_size)]
            pose_stack = [pose_seq[i+j] for j in range(stack_size)]

            image_paths_stacks.append(paths_stack)
            stamps_new.append(stamp_stack)
            poses_new.append(pose_stack)

            if not augment:
                continue

            if i % 20 == 0:
                paths_stack_new_2 = [paths_stack[0] for _ in range(stack_size)]
                pose_stack_new_2 = [pose_stack[0] for _ in range(stack_size)]

                image_paths_stacks.append(paths_stack_new_2)
                stamps_new.append(stamp_stack)
                poses_new.append(pose_stack_new_2)

    return image_paths_stacks, stamps_new, poses_new


def get_input_shape(image_paths):
    stack_size = len(image_paths[0])
    rows, cols, channels = cv2.imread(image_paths[0][0]).shape
    return rows, cols, stack_size, channels


def load_image_stacks(image_path_stacks, scalers=None):
    """Loads image path stacks into memory"""

    num_stacks = len(image_path_stacks)
    rows, cols, stack_size, channels = get_input_shape(image_path_stacks)

    shape = (num_stacks, rows, cols, stack_size, channels)

    image_stacks = np.zeros(shape, dtype=np.float32)

    paths = set([path for path_stack in image_path_stacks for path in path_stack])

    # Load all images into memory, flatten them, and normalize by channel
    images_flat = np.array([cv2.imread(path).flatten() for path in paths], dtype=np.float32)

    if scalers is None:
        scalers = [StandardScaler() for _ in range(channels)]
        for channel, scaler in enumerate(scalers):
            start, end = channel * rows * cols, (channel+1) * rows * cols
            channel_col = images_flat[:, start:end].reshape(-1, 1)
            scaler.fit(channel_col)
            channel_col = scaler.transform(channel_col)
            images_flat[:, start:end] = channel_col.reshape((images_flat.shape[0], rows * cols))
    else:
        for channel, scaler in enumerate(scalers):
            start, end = channel * rows * cols, (channel+1) * rows * cols
            channel_col = images_flat[:, start:end].reshape(-1, 1)
            channel_col = scaler.transform(channel_col)
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

    return image_stacks, scalers


def build_channel_model(channel_shape, stack_size):
    if YAW:
        regu = 0.005
        input_layer = Input(shape=channel_shape)
        hidden_layer = Conv3D(8, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Conv3D(16, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
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
    channel_shape[3] = 1

    r_input, r_output = build_channel_model(channel_shape, stack_size)
    g_input, g_output = build_channel_model(channel_shape, stack_size)
    b_input, b_output = build_channel_model(channel_shape, stack_size)

    output_avg = Average()([r_output, g_output, b_output])

    model = Model(inputs=[r_input, g_input, b_input], outputs=output_avg)

    return model


def load_data(data_dir, stack_size, load_data=True):

    images_train, stamps_train, poses_train = load_filenames_odom(data_dir, stack_size, TRAIN_SEQUENCES)
    images_train, stamps_train_stacks, poses_train_stacks = stack_data(images_train, stamps_train, poses_train, stack_size, True)

    images_test, stamps_test, poses_test = load_filenames_odom(data_dir, stack_size, TEST_SEQUENCES)
    images_test, stamps_test_stacks, poses_test_stacks = stack_data(images_test, stamps_test, poses_test, stack_size)

    input_shape = get_input_shape(images_train)

    odom_train = np.array([poses_to_offsets(p, ['yaw']) for p in poses_train_stacks])
    odom_test = np.array([poses_to_offsets(p, ['yaw']) for p in poses_test_stacks])

    if load_data:
        images_train, _ = load_image_stacks(images_train)
        images_test, _ = load_image_stacks(images_test)

        # Augment dataset 1 (flip)
        images_train_new = np.repeat(images_train, 2, axis=0)
        odom_train_new = np.repeat(odom_train, 2, axis=0)

        images_train_new[images_train.shape[0]:] = np.flip(images_train, axis=2)
        odom_train_new[odom_train.shape[0]:] = -odom_train

        assert images_train.shape[0] == odom_train.shape[0], '{} {}'.format(
            images_train.shape, odom_train.shape)

        images_train = images_train_new
        odom_train = odom_train_new
        # num_stacks = images_train[0].shape[0]

        # images_train_new = [np.repeat(s, 2, axis=0) for s in images_train]
        # odom_train_new = np.repeat(odom_train, 2, axis=0)

        # for i in range(3):
        #     images_train_new[i][num_stacks:] = np.flip(images_train[i], axis=2)
        
        # odom_train_new[num_stacks:] = -odom_train

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


def convert(image_stacks):
    image_stacks = [
        np.expand_dims(image_stacks[:, :, :, :, 0], axis=4),
        np.expand_dims(image_stacks[:, :, :, :, 1], axis=4),
        np.expand_dims(image_stacks[:, :, :, :, 2], axis=4)
    ]
    return image_stacks

def main(args):

    model_file_tpl = get_model_tpl(args.model_file)
    images_train, odom_train, images_test, odom_test, input_shape = load_data(
        args.data_dir, args.stack_size, args.preload)

    if args.resume:
        model = load_model(args.model_file, custom_objects={'weighted_mse': weighted_mse})
    else:
        model = build_model(input_shape, args.stack_size)
        # model.compile(loss='mse', optimizer='adam')
        model.compile(loss=weighted_mse, optimizer='adam')

    model.summary()

    callbacks = [ModelCheckpoint(model_file_tpl), RecentModelRenamer(args.model_file)]

    if args.preload:

        history = model.fit(convert(images_train), odom_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            shuffle=True,
            validation_data=(convert(images_test), odom_test),
            callbacks=callbacks)

    else:

        scalers = pickle.load(open(args.scalers, 'rb'))
        history = model.fit_generator(dataset_generator(images_train, odom_train, scalers, args.batch_size),
            epochs=args.epochs,
            steps_per_epoch=int(len(images_train)/args.batch_size),
            validation_steps=int(len(images_test)/args.batch_size),
            verbose=1,
            validation_data=dataset_generator(images_test, odom_test, scalers, args.batch_size),
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
    parser.add_argument('-p', '--preload', action='store_true', default=False, help='Load all data at once')
    parser.add_argument('-n', '--scalers', help='RGB scalers file')
    args = parser.parse_args()

    if not args.preload and not args.scalers:
        raise Exception('If --preload is not set --scalers must be')

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
