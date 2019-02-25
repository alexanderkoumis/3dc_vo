#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

import train



# stack_sizes = [3, 5, 7]
#stack_sizes = [2, 3, 5, 7, 10]
stack_sizes = [5]
# stack_sizes = [int(sys.argv[1])]
epochs = 600
epochs_save = 60

kitti_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# eval_bin = '/home/koumis/Development/External/kitty_eval/evaluate_odometry_quiet'
# data_dir = '/media/cache/koumis/kitti/odom/160_90'
eval_bin = sys.argv[1]
data_dir = sys.argv[2]

odo_gt_dir = os.path.join(data_dir, 'poses/')
odo_res_dir = os.path.join(data_dir, 'poses_results/')

model_dir = os.path.join(kitti_dir, 'models', 'odom')
results_dir = os.path.join(kitti_dir, 'results', 'odom')

train_file = os.path.join(kitti_dir, 'kitti_vo', 'train.py')
plot_file = os.path.join(kitti_dir, 'kitti_vo', 'plot.py')

create_results_script = os.path.join(kitti_dir, 'scripts', 'create_results_file_fixed_y.py')

subdata_dir = 'cool'
subdata_results_dir = os.path.join(odo_res_dir, subdata_dir, 'data')
plot_error_dir = os.path.join(odo_res_dir, subdata_dir, 'plot_error')
stats_file = os.path.join(odo_res_dir, subdata_dir, 'stats.txt')
output_file = os.path.join(odo_res_dir, subdata_dir, 'all_stack_results.txt')

shutil.rmtree(odo_res_dir)
os.mkdir(odo_res_dir)
os.makedirs(subdata_results_dir)


def get_model_file_epoch(model_stack_dir, stack_size, epoch):

    files = os.listdir(model_stack_dir)
    result = []

    for fname in files:

        if len(fname.split('.')) != 5:
            continue

        epoch_curr = int(fname.split('.')[1].split('-')[0])

        if epoch_curr == epoch:
            model_file_epoch = os.path.join(model_stack_dir, fname)
            val_loss = float(fname.split('.')[-3].split('-')[-1] + '.' + fname.split('.')[-2])
            result.append((model_file_epoch, val_loss, epoch))

    # print('Bummer dude model_stack_dir: {}, stack_size: {},  epoch: {}'.format(
    #     model_stack_dir, stack_size, epoch))

    return result


def create_results_file(model_file, stack_size, mode):

    create_results_command = '{} {} {} {} {} {}'.format(
        create_results_script, model_file, stack_size, data_dir, subdata_results_dir, mode)

    subprocess.check_call(create_results_command,
                          shell=True,
                          env={**os.environ, 'PYTHONPATH': os.path.join(kitti_dir, 'kitti_vo')},
                          stdout=open(os.devnull, 'wb'),
                          stderr=open(os.devnull, 'wb')
                          )


def train_model(model_file, history_file, stack_size):
    # train_command = f'{train_file} {data_dir} odom {model_file} {history_file} -m high -e {epochs} -b 100 -s {stack_size}'
    # print(f'Training stack size {stack_size} with command: {train_command}')
    # train_command = '{} {} {} {} -r -e {} -b 400 -s {}'.format(train_file, data_dir, model_file, history_file, epochs, stack_size)
    train_command = '{} {} {} {} -e {} -b 200 -s {}'.format(train_file, data_dir, model_file, history_file, epochs, stack_size)
    print('Training stack size {} with command: {}'.format(stack_size, train_command))
    subprocess.check_call(train_command,
                          shell=True,
                          env={**os.environ, 'PYTHONPATH': os.path.join(kitti_dir, 'kitti_vo')},
                          )
                          # stdout=open(os.devnull, 'wb'),
                          # stderr=open(os.devnull, 'wb'))

def is_float(num_str):
    try:
        float(num_str)
        return True
    except:
        return False

def eval_results():
    eval_results_cmd = '{} {}'.format(eval_bin, subdata_dir)
    # print('Running command {}'.format(eval_results_cmd))
    subprocess.check_call(eval_results_cmd,
                          shell=True,
                          env={**os.environ, 'ODO_RES_DIR': odo_res_dir, 'ODO_GT_DIR': odo_gt_dir},
                          stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

    def get_kitti_eval_avg(fname, scale=1.0):
        nums = []
        with open(fname, 'r') as fd:
            for line in fd.readlines():
                split_data = line.split()
                if len(split_data) == 2 and all(is_float(tok) for tok in split_data):
                    nums.append(float(split_data[1]) * scale)
        return statistics.mean(nums)

    tl_avgs = []
    rl_avgs = []
    for sequence in train.TEST_SEQUENCES:
        tl_file = os.path.join(plot_error_dir, '{}_tl.txt'.format(sequence))
        rl_file = os.path.join(plot_error_dir, '{}_rl.txt'.format(sequence))
        tl_avgs.append(get_kitti_eval_avg(tl_file))
        rl_avgs.append(get_kitti_eval_avg(rl_file, 57.2958))
    tl_avg = statistics.mean(tl_avgs)
    rl_avg = statistics.mean(rl_avgs)
    return tl_avg, rl_avg

def get_models_losses(model_stack_dir, stack_size):

    models_losses = []

    for epoch in range(1, epochs+1):
        models_losses += get_model_file_epoch(model_stack_dir, stack_size, epoch)

    models_losses.sort(key=lambda x: x[1])
    return models_losses


results = []


for stack_size in stack_sizes:

    model_stack_dir = os.path.join(model_dir, str(stack_size))
    model_file = os.path.join(model_stack_dir, 'model_odom.h5')
    history_file = os.path.join(results_dir, str(stack_size), 'history.json')

    # shutil.rmtree(model_stack_dir)
    # os.mkdir(model_stack_dir)
    train_model(model_file, history_file, stack_size)

    models_losses = get_models_losses(model_stack_dir, stack_size)

    for model_file_epoch, val_loss, epoch in models_losses[:epochs_save]:
        # for mode in ['normal', 'flipped', 'merged']:
        for mode in ['normal', 'flipped']:
            create_results_file(model_file_epoch, stack_size, mode)
            trans_err, rot_err = eval_results()
            result_tup = (trans_err, rot_err, stack_size, epoch, model_file_epoch, val_loss, mode)
            results.append(result_tup)

            print('val_loss: {}, epoch: {}, mode: {} trans_err: {}, rot_err: {}'.format(
                val_loss, epoch, mode, trans_err, rot_err))


results.sort()
result_str = json.dumps(results)
with open(output_file, 'w+') as fd:
    fd.write(result_str)

def print_results(r):
    for result in r:
        print('trans_err: {}, rot_err: {}, stack_size: {}, epoch: {}, mode: {}, val_loss: {}'.format(
            result[0], result[1], result[2], result[3], result[6], result[5]))


print('\n\n\n')

print_results(results)

print('\n\n\n')

results.sort(key=lambda x: x[1])

print_results(results)


