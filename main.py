#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import RMSprop
# from keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam


def load_image(image_path, image_name):
    full_path = os.path.join(image_path, f'{image_name}.png')
    image = cv2.imread(full_path)
    # image = image / np.max(image)
    # image -= np.mean(image)
    return image

def parse_ground_truth(gt_path, image_name):
    full_path = os.path.join(gt_path, f'{image_name}.txt')
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
    stacked_images = np.zeros((len(image_data)-stack_size+1, rows, cols, stack_channels))
    # (478, 375, 1242, 12)
    for i in range(len(image_data)-stack_size+1):
        for j in range(stack_size):
            stacked_images[i, :, :, j*channels:(j+1)*channels] = image_data[i+j]
    return stacked_images


image_path = '/Users/alexander/Development/KITTI/2011_09_26_drive_0019_sync/image_02/data'
gt_path = '/Users/alexander/Development/KITTI/2011_09_26_drive_0019_sync/oxts/data'

image_paths = os.listdir(image_path)
image_names = [path.split('.')[0] for path in image_paths]
image_names.sort(key=int)

image_data = [load_image(image_path, name) for name in image_names]
gt_data = [parse_ground_truth(gt_path, name) for name in image_names]

stack_size = 4
# image_data = np.array(stack_images(image_data, stack_size))
image_data = stack_images(image_data, stack_size)
gt_data = np.array(gt_data[:-stack_size+1])


X_train, X_test, y_train, y_test = train_test_split(image_data, gt_data, test_size=1.0/4.0)


print('Splitted up, creating model')
num_images, image_rows, image_cols, image_channels = image_data.shape
num_outputs = gt_data.shape[1]

input_shape = image_rows, image_cols, image_channels

# https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(4, 4), padding='same', activation='linear', input_shape=input_shape))
model.add(LeakyReLU())
model.add(Conv2D(32, (5, 5), strides=(4, 4), padding='same', activation='linear'))
model.add(LeakyReLU())
model.add(Conv2D(16, (5, 5), strides=(4, 4), padding='same', activation='linear'))
model.add(LeakyReLU())
model.add(Conv2D(8, (5, 5), strides=(4, 4), padding='same', activation='linear'))
model.add(LeakyReLU())
model.add(Flatten())
model.add(Dense(num_outputs, activation='sigmoid'))

model.summary()

print('Model made, compiling')
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

print('Fitting')
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=100000,
                    verbose=1,
                    validation_data=(X_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)