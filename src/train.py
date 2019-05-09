#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv3D, Flatten, Dropout, BatchNormalization, LeakyReLU, Input, Average
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

from recent_model_renamer import RecentModelRenamer
import common


def poses_to_offsets(pose_stack, info_requested):

    def yaw_from_matrix(M):
        cy = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
        yaw = math.atan2(-M[2, 0],  cy)
        return yaw

    first_pose, last_pose = pose_stack[0], pose_stack[-1]

    R_first, R_last = first_pose[:3, :3], last_pose[:3, :3]
    t_first, t_last = first_pose[:3, 3], last_pose[:3, 3]

    R_diff = R_last.T.dot(R_first)
    t_diff = R_first.T.dot(t_last - t_first)
    x_diff, z_diff, y_diff = t_diff

    yaw_diff = yaw_from_matrix(R_diff.T)

    info = {
        'x': x_diff,
        'y': y_diff,
        'z': z_diff,
        'yaw': yaw_diff
    }

    result = np.array([info[val] for val in info_requested])

    return result


def build_channel_model(channel_shape, yaw_or_y, stack_size):
    if yaw_or_y == common.yaw:
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
        hidden_layer = Conv3D(8, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)
        hidden_layer = Conv3D(8, [3, 3, stack_size], strides=[3, 3, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dropout(0.1)(hidden_layer)
        hidden_layer = Conv3D(16, [1, 1, 1], strides=[1, 1, 1], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Conv3D(1, [1, 1, stack_size], strides=[1, 1, stack_size], kernel_regularizer=l2(regu), padding='same', activation='relu')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Flatten()(hidden_layer)
        hidden_layer = Dense(8, kernel_regularizer=l2(regu), activation='relu')(hidden_layer)
        hidden_layer = LeakyReLU()(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)
        output_layer = Dense(1, activation='linear')(hidden_layer)
    return input_layer, output_layer


def build_model(input_shape, yaw_or_y, stack_size):

    channel_shape = list(copy.deepcopy(input_shape))
    channel_shape[3] = 1

    r_input, r_output = build_channel_model(channel_shape, yaw_or_y, stack_size)
    g_input, g_output = build_channel_model(channel_shape, yaw_or_y, stack_size)
    b_input, b_output = build_channel_model(channel_shape, yaw_or_y, stack_size)

    output_avg = Average()([r_output, g_output, b_output])

    model = Model(inputs=[r_input, g_input, b_input], outputs=output_avg)

    return model


def load_data(data_dir, stack_size, yaw_or_y):

    images_train, stamps_train, poses_train = common.load_filenames_odom(data_dir, common.sequences_train)
    images_train, stamps_train_stacks, poses_train_stacks = common.stack_data(images_train, stamps_train, poses_train, stack_size)

    images_val, stamps_val, poses_val = common.load_filenames_odom(data_dir, common.sequences_val)
    images_val, stamps_val_stacks, poses_val_stacks = common.stack_data(images_val, stamps_val, poses_val, stack_size)

    input_shape = common.get_input_shape(images_train)

    odom_train = np.array([poses_to_offsets(p, [yaw_or_y]) for p in poses_train_stacks])
    odom_val = np.array([poses_to_offsets(p, [yaw_or_y]) for p in poses_val_stacks])

    images_train = common.load_image_stacks(images_train)
    images_val = common.load_image_stacks(images_val)

    if yaw_or_y == common.yaw:
        # If we are training the rotation network (yaw), augment the dataset
        # by flipping the images and yaw values
        num_stacks = images_train.shape[0]
        images_train = np.repeat(images_train, 2, axis=0)
        odom_train = np.repeat(odom_train, 2, axis=0)

        images_train[num_stacks:] = np.flip(images_train[:num_stacks], axis=2)
        odom_train[num_stacks:] = -odom_train[:num_stacks]

    return images_train, odom_train, images_val, odom_val, input_shape


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


def main(args):

    model_file_tpl = get_model_tpl(args.model_file)
    images_train, odom_train, images_val, odom_val, input_shape = load_data(
        args.data_dir, args.stack_size, args.yaw_or_y)

    if args.resume:
        model = load_model(args.model_file, custom_objects={'weighted_mse': common.weighted_mse})
    else:
        model = build_model(input_shape, args.yaw_or_y, args.stack_size)
        model.compile(loss=common.weighted_mse if args.yaw_or_y == common.yaw else 'mse',
                      optimizer=Adam(lr=0.0001))

    model.summary()

    callbacks = [ModelCheckpoint(model_file_tpl), RecentModelRenamer(args.model_file)]

    history = model.fit(common.split_image_channels(images_train), odom_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        shuffle=True,
        validation_data=(common.split_image_channels(images_val), odom_val),
        callbacks=callbacks)

    print('Saving model to {}'.format(args.model_file))
    model.save(args.model_file)

    history_file = args.model_file + '.history'
    print('Saving history to {}'.format(history_file))
    save_history_file(history_file + '.history', history.history)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Dataset directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('yaw_or_y', choices=[common.yaw, common.y], help='Train yaw or y network')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('-s', '--stack_size', type=int, default=common.stack_size, help='Size of image stack')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume training on model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
