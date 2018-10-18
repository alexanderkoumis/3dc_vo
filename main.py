#!/usr/bin/env python3

import argparse
import os
import sys

import cv2
import numpy as np

from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='Image path')
    parser.add_argument('odometry_dir', help='Odometry ground truth path')
    parser.add_argument('model_file', help='Where to save/load model')
    parser.add_argument('-s', '--stack_size', default=4, help='Size of image stack')
    parser.add_argument('-e', '--test', action='store_true', help='Test saved model')
    args = parser.parse_args()
    return args

def load_image(image_path, image_name):
    """Load and regularize image"""
    full_path = os.path.join(image_path, '{}.png'.format(image_name))
    image = cv2.imread(full_path)
    image = image / np.max(image)
    image -= np.mean(image)
    return image

def load_odometry(odom_path, image_name):
    """Load target odometry"""
    full_path = os.path.join(odom_path, '{}.txt'.format(image_name))
    with open(full_path, 'r') as fd:
        # 8  vf:    forward velocity, i.e. parallel to earth-surface (m/s)
        # 9  vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
        # 10 vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
        # 14 af:    forward acceleration (m/s^2)
        # 15 al:    leftward acceleration (m/s^2)
        # 16 au:    upward acceleration (m/s^2)
        # 20 wf:    angular rate around forward axis (rad/s)
        # 21 wl:    angular rate around leftward axis (rad/s)
        # 22 wu:    angular rate around upward axis (rad/s)
        data = fd.read().split()
        idxs = [8, 9, 10, 14, 15, 16, 20, 21, 22]
        vals = [float(data[idx]) for idx in idxs]
        return vals

def stack_images(image_data, stack_size):
    rows, cols, channels = image_data[0].shape
    stack_channels = channels * stack_size
    stacked_images = np.zeros((len(image_data)-stack_size+1, rows, cols, stack_channels), dtype=np.float32)
    for i in range(len(image_data)-stack_size+1):
        for j in range(stack_size):
            stacked_images[i, :, :, j*channels:(j+1)*channels] = image_data[i+j]
    return stacked_images

def load_data(image_dir, odom_dir, stack_size):
    image_paths = os.listdir(image_dir)
    image_names = [path.split('.')[0] for path in image_paths]
    image_names.sort(key=int)

    image_data = [load_image(image_dir, name) for name in image_names]
    odom_data = [load_odometry(odom_dir, name) for name in image_names]

    # Stack images, trim last few odometries
    image_data = stack_images(image_data, stack_size)
    odom_data = np.array(odom_data[:-stack_size+1])

    return image_data, odom_data

def build_model(input_shape, num_outputs):
    # https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(4, 4), padding='same', activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Conv2D(8, (5, 5), strides=(4, 4), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Conv2D(4, (5, 5), strides=(4, 4), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='relu'))
    return model

def evaluate_model(model, features, labels):
    loss, accuracy = model.evaluate(features, labels, verbose=1)
    print('Final loss: {}, accuracy: {}'.format(loss, accuracy))

def test_saved_model(features, labels, model_file):
    model = load_model(model_file)
    evaluate_model(model, features, labels)

def main(args):

    image_data, odom_data = load_data(args.image_dir, args.odometry_dir, args.stack_size)
    X_train, X_test, y_train, y_test = train_test_split(image_data, odom_data, test_size=1.0/4.0)

    if args.test:
        print('Testing saved model {}'.format(args.model_file))
        test_saved_model(X_test, y_test, args.model_file)
        sys.exit()

    num_images, image_rows, image_cols, image_channels = image_data.shape
    input_shape = (image_rows, image_cols, image_channels)
    num_outputs = odom_data.shape[1]

    model = build_model(input_shape, num_outputs)

    model.summary()
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=4,
                        epochs=50,
                        verbose=1,
                        validation_data=(X_test, y_test))

    evaluate_model(model, X_test, y_test)

    model.save(args.model_file)

if __name__ == '__main__':
    main(parse_args())
