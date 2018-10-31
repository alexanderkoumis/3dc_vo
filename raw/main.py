#!/usr/bin/env python3

import argparse
import os
import random
import sys
from os.path import join

import cv2
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from filename_loaders import load_filenames_raw, load_filenames_odom
from image_loader import ImageLoader


DEFAULT_STACK_SIZE = 3


def dataset_generator(image_paths_raw, odom_raw, batch_size, train=True):

    paths_train, paths_test, odom_train, odom_test = train_test_split(image_paths_raw, odom_raw)
    image_paths_all, odom_all = (paths_train, odom_train) if train else (paths_test, odom_test)
    image_loader = ImageLoader()

    while True:

        stacked_images_batch = []
        odom_batch = []

        for image_paths, odom in zip(image_paths_all, odom_all):

            images = [image_loader.load_image(path) / 255.0 for path in image_paths]
            stacked_images = np.dstack(images).astype(float)
            stacked_images_batch.append(stacked_images)
            odom_batch.append(odom)

            if len(stacked_images_batch) == batch_size:
                yield np.array(stacked_images_batch), np.array(odom_batch)
                stacked_images_batch = []
                odom_batch = []


def stack_data(image_paths, stamps, odom, stack_size):
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
    stamps_stacks = []
    odom_stacks = []

    for image_paths_seq, stamps_seq, odom_seq in zip(image_paths, stamps, odom):

        for i in range(len(image_paths_seq)-stack_size+1):

            image_paths_stack = [image_paths_seq[i+j] for j in range(stack_size)]
            stamps_stack = [stamps_seq[i+j] for j in range(stack_size)]
            odom_stack = [odom_seq[i+j] for j in range(stack_size)]

            image_paths_stacks.append(image_paths_stack)
            stamps_stacks.append(stamps_stack)
            odom_stacks.append(odom_stack)

    return image_paths_stacks, stamps_stacks, odom_stacks


def load_filenames(base_dir, dataset_type):
    loader = {
        'raw': load_filenames_raw,
        'odom': load_filenames_odom
    }
    return loader[dataset_type](base_dir)


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


def get_input_shape(image_paths, stack_size):
    image = cv2.imread(image_paths[0][0])
    return image.shape * np.array([1, 1, stack_size])


def evaluate_model(model, image_paths, odom, batch_size):
    loss, accuracy = model.evaluate_generator(
        dataset_generator(image_paths, odom, batch_size, False))
    print('Final loss: {}, accuracy: {}'.format(loss, accuracy))


def test_saved_model(model_file, image_paths, odom, batch_size):
    model = load_model(model_file)
    evaluate_model(model, image_paths, odom, batch_size)


def main(args):

    image_paths, stamps, odom, num_outputs = load_filenames(args.base_dir, args.dataset_type)
    image_paths, stamps, odom = stack_data(image_paths, stamps, odom, args.stack_size)

    if args.evaluate:
        print('Testing saved model {}'.format(args.model_file))
        test_saved_model(args.model_file, image_paths, odom, args.batch_size)
        sys.exit()

    input_shape = get_input_shape(image_paths, args.stack_size)
    model = build_model(input_shape, num_outputs)

    optimizer = SGD(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    model.fit_generator(dataset_generator(image_paths, odom, args.batch_size, True),
                        epochs=2,
                        steps_per_epoch=int(0.75*(len(image_paths)/args.batch_size)),
                        validation_steps=int(0.25*(len(image_paths)/args.batch_size)),
                        verbose=1,
                        validation_data=dataset_generator(image_paths, odom, args.batch_size, False))

    model.save(args.model_file)

    print('Saving model to {}'.format(args.model_file))
    model.save(args.model_file)

    evaluate_model(model, image_paths, odom, args.batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Base directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('dataset_type', help='Dataset type (either raw or odom)')
    parser.add_argument('-b', '--batch_size', default=10, help='Batch size')
    parser.add_argument('-s', '--stack_size', default=DEFAULT_STACK_SIZE, help='Size of image stack')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Test saved model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
