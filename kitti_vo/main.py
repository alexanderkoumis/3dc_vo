#!/usr/bin/env python3

import argparse
import random
import sys

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from filename_loaders import load_filenames_raw, load_filenames_odom
from image_loader import ImageLoader


def dataset_generator(image_paths_all, odom_all, batch_size, memory):

    image_loader = ImageLoader(memory)

    input_target_list = list(zip(image_paths_all, odom_all))
    random.shuffle(input_target_list)

    while True:

        stacked_images_batch = []
        odom_batch = []

        for image_paths, odom in input_target_list:

            images = [image_loader.load_image(path)/255.0 for path in image_paths]
            stacked_images = np.dstack(images)
            stacked_images_batch.append(stacked_images)
            odom_batch.append(odom)

            if len(stacked_images_batch) == batch_size:
                yield np.array(stacked_images_batch), np.array(odom_batch)
                stacked_images_batch = []
                odom_batch = []


def stack_data(image_paths, stamps, odoms, stack_size):
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

    high_low_ratio = 0.5
    high_angle_thresh = 0.13
    high_angle_count = sum(abs(odom) > high_angle_thresh for odom in odoms_new)
    low_angle_keep = int(high_angle_count * high_low_ratio)

    image_paths_stacks_new_new = []
    stamps_new_new = []
    odoms_new_new = []
    idxs = []

    for idx, (path_stack, stamp, odom) in enumerate(zip(image_paths_stacks, stamps_new, odoms_new)):
        if abs(odom) > high_angle_thresh:
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

    high_angle_count = sum(abs(odom) > high_angle_thresh for odom in odoms_new_new)
    low_angle_count = sum(abs(odom) < high_angle_thresh for odom in odoms_new_new)

    return image_paths_stacks_new_new, stamps_new_new, np.array(odoms_new_new)


def get_input_shape(image_paths, stack_size):
    image = cv2.imread(image_paths[0][0])
    return image.shape * np.array([1, 1, stack_size])


def load_filenames(data_dir, dataset_type, stack_size):
    loader = {
        'raw': load_filenames_raw,
        'odom': load_filenames_odom
    }
    return loader[dataset_type](data_dir, stack_size)


def load_image_stacks(image_path_stacks):
    """Loads image path stacks into memory"""
    num_stacks = len(image_path_stacks)
    stack_size = len(image_path_stacks[0])
    rows, cols, channels = get_input_shape(image_path_stacks, stack_size)
    shape = (num_stacks, rows, cols, channels)

    image_stacks = np.zeros(shape, dtype=np.float32)

    paths = set([path for path_stack in image_path_stacks for path in path_stack])

    # Load all images into memory, flatten them, and normalize by channel
    images_flat = np.array([cv2.imread(path).flatten()/255.0 for path in paths], dtype=np.float32)
    for channel in range(3):
        start, end = channel * rows * cols, (channel+1) * rows * cols
        images_flat[:, start:end] = StandardScaler().fit_transform(images_flat[:, start:end])

    image_cache = {path: img.reshape((rows, cols, 3)) for path, img in zip(paths, images_flat)}

    for idx, path_stack in enumerate(image_path_stacks):
        image_stack = np.dstack(image_cache[path] for path in path_stack)
        image_stacks[idx] = image_stack

    return image_stacks


def build_model(input_shape, num_outputs):
    model = Sequential()
    model.add(Conv2D(64, (4, 4), strides=(4, 4), padding='same', kernel_regularizer=l2(0.00), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.00)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.00)))
    model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.00)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(num_outputs, activation='linear'))
    return model


def main(args):

    image_paths, stamps, odom, num_outputs = load_filenames(args.data_dir, args.dataset_type, args.stack_size)
    image_paths, stamps, odom = stack_data(image_paths, stamps, odom, args.stack_size)

    if args.resume:
        model = load_model(args.model_file)
    else:
        input_shape = get_input_shape(image_paths, args.stack_size)
        model = build_model(input_shape, num_outputs)
        model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()

    if args.memory == 'high':
        image_stacks = load_image_stacks(image_paths)
        images_train, images_test, odom_train, odom_test = train_test_split(image_stacks, odom)

        model.fit(images_train, odom_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            validation_data=(images_test, odom_test),
            callbacks=[ModelCheckpoint(args.model_file)])
    else:
        num_batches = len(image_paths) / args.batch_size
        paths_train, paths_test, odom_train, odom_test = train_test_split(image_paths, odom)

        model.fit_generator(dataset_generator(paths_train, odom_train, args.batch_size, args.memory),
            epochs=args.epochs,
            steps_per_epoch=int(0.75*num_batches),
            validation_steps=int(0.25*num_batches),
            verbose=1,
            validation_data=dataset_generator(paths_test, odom_test, args.batch_size, args.memory),
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
