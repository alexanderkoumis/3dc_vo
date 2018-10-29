#!/usr/bin/env python3

import argparse
import os
import random

import cv2
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.regularizers import l2


DEFAULT_STACK_SIZE = 3


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


def get_stamps(stamps_path):
    result = []
    with open(stamps_path, 'r') as fd:
        for line in fd.readlines():
            result.append(float(line))
    return result


def get_poses(poses_path):
    result = []
    with open(poses_path, 'r') as fd:
        for line in fd.readlines():
            pose = np.fromstring(line, dtype=float, sep=' ')
            pose = pose.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            result.append(pose)
    return result


def load_filenames(base_dir):

    image_paths_all = []
    stamps_all = []
    poses_all = []

    pose_dir = os.path.join(base_dir, 'poses')
    sequences_dir = os.path.join(base_dir, 'sequences')

    sequence_nums = os.listdir(sequences_dir)
    sequence_nums.sort(key=int)

    for sequence_num in sequence_nums:

        # Only sequences 0-10 are provided for ground truth
        if int(sequence_num) > 10:
            break

        image_dir = os.path.join(sequences_dir, sequence_num, 'image_2')
        stamps_path = os.path.join(sequences_dir, sequence_num, 'times.txt')
        pose_path = os.path.join(pose_dir, '{}.txt'.format(sequence_num))

        image_filenames = os.listdir(image_dir)
        image_filenames.sort(key=lambda x: int(x.split('.')[0]))

        image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
        stamps = get_stamps(stamps_path)
        poses = get_poses(pose_path)

        assert len(image_paths) == len(stamps) == len(poses), '{} {} {}'.format(
            len(image_paths), len(stamps), len(poses))

        image_paths_all.append(image_paths)
        stamps_all.append(stamps)
        poses_all.append(poses)

    return image_paths_all, stamps_all, poses_all


def stack_data(image_paths, stamps, poses, stack_size):
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
    poses_stacks = []

    for image_paths_seq, stamps_seq, poses_seq in zip(image_paths, stamps, poses):

        for i in range(len(image_paths_seq)-stack_size+1):

            image_paths_stack = [image_paths_seq[i+j] for j in range(stack_size)]
            stamps_stack = [stamps_seq[i+j] for j in range(stack_size)]
            poses_stack = [poses_seq[i+j] for j in range(stack_size)]

            image_paths_stacks.append(image_paths_stack)
            stamps_stacks.append(stamps_stack)
            poses_stacks.append(poses_stack)

    return image_paths_stacks, stamps_stacks, poses_stacks


def test_train_split(image_paths, stamps, poses, ratio=1.0/4.0):

    num_stacks = len(image_paths)

    test_size = int(num_stacks*ratio)
    train_size = num_stacks - test_size

    train_idxs = set(random.sample(range(num_stacks), train_size))
    test_idxs = set(range(num_stacks)) - train_idxs

    image_paths_train = [image_paths[i] for i in train_idxs]
    stamps_train = [stamps[i] for i in train_idxs]
    poses_train = [poses[i] for i in train_idxs]

    image_paths_test = [image_paths[i] for i in test_idxs]
    stamps_test = [stamps[i] for i in test_idxs]
    poses_test = [poses[i] for i in test_idxs]

    train_data = (image_paths_train, stamps_train, poses_train)
    test_data = (image_paths_test, stamps_test, poses_test)

    return train_data, test_data


def calc_velocity(stamps, poses, scale_data=None):

    first_stamp, last_stamp = stamps[0], stamps[-1]
    first_pose, last_pose = poses[0], poses[-1]

    time_elapsed = last_stamp - first_stamp
    transform_world = np.linalg.inv(first_pose).dot(last_pose)
    R_world, t_world = transform_world[:3, :3], transform_world[:3, 3]

    t_cam = -R_world.T.dot(t_world)

    velocity = (t_cam / time_elapsed)[:2]

    return velocity


def dataset_generator(image_paths_raw, stamps_raw, poses_raw, batch_size, train=True):

    train_data, test_data = test_train_split(image_paths_raw, stamps_raw, poses_raw)
    image_paths_all, stamps_all, poses_all = train_data if train else test_data
    image_loader = ImageLoader()

    while True:

        stacked_images_all = []
        velocity_all = []

        for image_paths, stamps, poses in zip(image_paths_all, stamps_all, poses_all):

            velocity = calc_velocity(stamps, poses)
            images = [image_loader.load_image(path) / 255.0 for path in image_paths]
            stacked_images = np.dstack(images).astype(float)

            stacked_images_all.append(stacked_images)
            velocity_all.append(velocity)

            if len(stacked_images_all) == batch_size:
                yield np.array(stacked_images_all), np.array(velocity_all)
                stacked_images_all = []
                velocity_all = []


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


def main(args):
    image_paths, stamps, poses = load_filenames(args.base_dir)
    image_paths, stamps, poses = stack_data(image_paths, stamps, poses, args.stack_size)

    # Just doing translational velocity for now
    num_outputs = 2

    input_shape = get_input_shape(image_paths, args.stack_size)
    model = build_model(input_shape, num_outputs)

    optimizer = SGD(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(dataset_generator(image_paths, stamps, poses, args.batch_size, True),
                                  epochs=2,
                                  steps_per_epoch=int(0.75*(len(image_paths)/args.batch_size)),
                                  validation_steps=int(0.25*(len(image_paths)/args.batch_size)),
                                  verbose=1,
                                  validation_data=dataset_generator(image_paths, stamps, poses, args.batch_size, False))

    model.save(args.model_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help='Base directory')
    parser.add_argument('model_file', help='Model file')
    parser.add_argument('-b', '--batch_size', help='Batch size', default=10)
    parser.add_argument('-s', '--stack_size', help='Size of image stack', default=DEFAULT_STACK_SIZE)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
