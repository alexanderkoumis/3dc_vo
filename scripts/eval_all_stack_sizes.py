#!/usr/bin/env python3.6

import os
import subprocess
import sys


stack_sizes = [3, 5, 7, 10, 20]


kitti_dir = '/home/koumis/Development/kitti_vo'
model_dir = os.path.join(kitti_dir, 'models', 'odom')
results_dir = os.path.join(kitti_dir, 'results', 'odom')
data_dir = '/media/cache/koumis/kitti/odom/160_90'

main_file = os.path.join(kitti_dir, 'kitti_vo', 'main.py')
plot_file = os.path.join(kitti_dir, 'kitti_vo', 'plot.py')


for stack_size in stack_sizes:

    model_stack_dir = os.path.join(model_dir, str(stack_size))
    results_stack_dir = os.path.join(model_dir, str(stack_size))

    model_file = os.path.join(model_stack_dir, 'model_odom.h5')

    train_command = f'{main_file} {data_dir} {model_file} odom -m high -e 200 -b 100 -s {stack_size}'
    plot_command = f'{plot_file} {stack_size}'

    print(f'Training stack size {stack_size}')
    subprocess.check_call(train_command, shell=True)

    print(f'Plotting stack size {stack_size}')
    subprocess.check_call(plot_command, shell=True)
