#!/usr/bin/env python3

import argparse
import random
import sys

import cv2
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from filename_loaders import load_filenames_raw, load_filenames_odom
from image_loader import ImageLoader


DEFAULT_STACK_SIZE = 5


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

    return image_paths_stacks, stamps_new, np.array(odoms_new)


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

    num_stacks = len(image_path_stacks)
    stack_size = len(image_path_stacks[0])
    rows, cols, channels = get_input_shape(image_path_stacks, stack_size)
    shape = (num_stacks, rows, cols, channels)

    image_stacks = np.zeros(shape, dtype=np.float32)

    for idx, path_stack in enumerate(image_path_stacks):
        image_stack = np.dstack(cv2.imread(path) for path in path_stack) / 255.0
        image_stacks[idx] = image_stack / 255.0

    return image_stacks


def build_model(input_shape, num_outputs):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(5, 5), padding='same', kernel_regularizer=l2(0.00), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(8, (5, 5), strides=(5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.00)))
    model.add(MaxPooling2D())
    model.add(Conv2D(8, (5, 5), strides=(5, 5), padding='same', kernel_regularizer=l2(0.00)))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_outputs, activation='linear'))
    return model


def evaluate_model(model, image_paths, odom, batch_size):
    model = load_model(model_file)
    num_batches = len(image_paths) / batch_size
    loss, accuracy = model.evaluate_generator(
        dataset_generator(image_paths, odom, batch_size, False), steps=num_batches)
    print('Final loss: {}, accuracy: {}'.format(loss, accuracy))


def main(args):

    image_paths, stamps, odom, num_outputs = load_filenames(args.data_dir, args.dataset_type, args.stack_size)
    image_paths, stamps, odom = stack_data(image_paths, stamps, odom, args.stack_size)
    num_batches = len(image_paths) / args.batch_size

    if args.test:
        print('Testing saved model {}'.format(args.model_file))
        evaluate_model(args.model_file, image_paths, odom, args.batch_size)
        sys.exit()

    paths_train, paths_test, odom_train, odom_test = train_test_split(image_paths, odom)

    input_shape = get_input_shape(image_paths, args.stack_size)
    model = build_model(input_shape, num_outputs)

    optimizer = SGD(lr=0.01, clipnorm=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    model.fit_generator(dataset_generator(paths_train, odom_train, args.batch_size, args.memory),
        epochs=args.epochs,
        steps_per_epoch=int(0.75*num_batches),
        validation_steps=int(0.25*num_batches),
        verbose=1,
        validation_data=dataset_generator(paths_test, odom_test, args.batch_size, args.memory))

    print('Saving model to {}'.format(args.model_file))
    model.save(args.model_file)


def main_high_mem(args):
    image_paths, stamps, odom, num_outputs = load_filenames(args.data_dir, args.dataset_type, args.stack_size)
    image_paths, stamps, odom = stack_data(image_paths, stamps, odom, args.stack_size)

    image_stacks = load_image_stacks(image_paths)

    images_train, images_test, odom_train, odom_test = train_test_split(image_stacks, odom)

    input_shape = get_input_shape(image_paths, args.stack_size)
    model = build_model(input_shape, num_outputs)

    optimizer = SGD(lr=0.01, clipnorm=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    model.fit(images_train, odom_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=1.0/4.0,
        verbose=1,
        validation_data=(images_test, odom_test))

    print('Saving model to {}'.format(args.model_file))
    model.save(args.model_file)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Dataset directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('dataset_type', help='Dataset type (either raw or odom)')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('-s', '--stack_size', default=DEFAULT_STACK_SIZE, help='Size of image stack')
    parser.add_argument('-t', '--test', action='store_true', help='Test saved model')
    parser.add_argument('-m', '--memory', default='low', help='Memory strategy, one of (low, medium, high)')
    args = parser.parse_args()

    if args.memory not in {'low', 'medium', 'high'}:
        sys.exit('--memory option must be one of low, medium, or high')

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.memory == 'high':
        main_high_mem(args)
    else:
        main(args)
