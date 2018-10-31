#!/usr/bin/env python3

import argparse
import os
import random
import sys
from os.path import join
from dateutil.parser import parser

import cv2
import numpy as np

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

"""
8  vf:    forward velocity, i.e. parallel to earth-surface (m/s)
9  vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
10 vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
14 af:    forward acceleration (m/s^2)
15 al:    leftward acceleration (m/s^2)
16 au:    upward acceleration (m/s^2)
20 wf:    angular rate around forward axis (rad/s)
21 wl:    angular rate around leftward axis (rad/s)
22 wu:    angular rate around upward axis (rad/s)
"""

DEFAULT_STACK_SIZE = 3
ODOM_IDXS = [8, 9, 21, 22]


class ImageLoader(object):

    def __init__(self, max_images=3000):
        self.cache = {}
        self.max_images = max_images

    def load_image(self, image_path):
        if image_path not in self.cache:
            if len(self.cache) == self.max_images:
                self.delete_random_image()
            image = cv2.imread(image_path)
            self.cache[image_path] = image
        return self.cache[image_path]

    def delete_random_image(self):
        del_idx = np.random.randint(self.max_images)
        del_key = list(self.cache.keys())[del_idx]
        del self.cache[del_key]


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


def load_odometry(odom_path, image_name):
    """Load target odometry"""
    full_path = join(odom_path, '{}.txt'.format(image_name))
    with open(full_path, 'r') as fd:
        data = fd.read().split()
        vals = [float(data[idx]) for idx in ODOM_IDXS]
        return vals


def load_filenames(base_dir):

    """
    kitti_processed/
        2011_09_26_drive_0001_sync/
            image_02/
                timestamps.txt
                data/
                    [d+].png
            oxts/
                dataformat.txt
                timestamps.txt
                data/
                    [d+].txt
    """
    image_paths_all = []
    odom_all = []
    stamps_all = []

    sequences = os.listdir(base_dir)

    for sequence in sequences:

        sequence_dir = join(base_dir, sequence)

        image_data_dir = join(sequence_dir, 'image_02', 'data')
        oxts_dir = join(sequence_dir, 'oxts')
        oxts_data_dir = join(oxts_dir, 'data')

        oxts_stamps_txt = join(oxts_dir, 'timestamps.txt')

        image_fnames = os.listdir(image_data_dir)
        image_fnames.sort(key=lambda x: int(x.split('.')[0]))
        image_names = [path.split('.')[0] for path in image_fnames]

        image_full_fnames = [join(image_data_dir, fname) for fname in image_fnames]
        odom_data = [load_odometry(oxts_data_dir, name) for name in image_names]
        stamps = [parser().parse(val).timestamp() for val in open(oxts_stamps_txt).readlines()]

        image_paths_all.append(image_full_fnames)
        odom_all.append(odom_data)
        stamps_all.append(stamps)

    return image_paths_all, odom_all, stamps_all



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

    image_paths, stamps, odom = load_filenames(args.base_dir)
    image_paths, stamps, odom = stack_data(image_paths, stamps, odom, args.stack_size)

    if args.test:
        print('Testing saved model {}'.format(args.model_file))
        test_saved_model(args.model_file, image_paths, odom, args.batch_size)
        sys.exit()

    input_shape = get_input_shape(image_paths, args.stack_size)
    num_outputs = len(ODOM_IDXS)
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
    parser.add_argument('-b', '--batch_size', help='Batch size', default=10)
    parser.add_argument('-s', '--stack_size', help='Size of image stack', default=DEFAULT_STACK_SIZE)
    parser.add_argument('-e', '--test', action='store_true', help='Test saved model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
