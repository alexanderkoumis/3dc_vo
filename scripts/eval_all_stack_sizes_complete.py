#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys



# stack_sizes = [3, 5, 7]
# stack_sizes = [3]
stack_sizes = [7]
epochs = 150
epochs_save = 20

kitti_dir = '/home/koumis/Development/kitti_vo'
eval_bin = '/home/koumis/Development/External/kitty_eval/evaluate_odometry_quiet'
# data_dir = '/media/cache/koumis/kitti/odom/160_30'
data_dir = '/media/cache/koumis/kitti/odom/160_90'

preload_model = '/home/koumis/Development/kitti_vo/models/odom/5/model_odom_yaw.h5'

sys.path.append(os.path.join(kitti_dir, 'kitti_vo'))

odo_gt_dir = os.path.join(data_dir, 'poses/')
odo_res_dir = os.path.join(data_dir, 'poses_results/')

model_dir = os.path.join(kitti_dir, 'models', 'odom')
results_dir = os.path.join(kitti_dir, 'results', 'odom')

train_file = os.path.join(kitti_dir, 'kitti_vo', 'train.py')
plot_file = os.path.join(kitti_dir, 'kitti_vo', 'plot.py')

create_results_script = os.path.join(kitti_dir, 'scripts', 'create_results_file.py')

subdata_dir = 'cool'
subdata_results_dir = os.path.join(odo_res_dir, subdata_dir, 'data')
stats_file = os.path.join(odo_res_dir, subdata_dir, 'stats.txt')
output_file = os.path.join(odo_res_dir, subdata_dir, 'all_stack_results.txt')

shutil.rmtree(odo_res_dir)
os.mkdir(odo_res_dir)
os.makedirs(subdata_results_dir)


def get_model_file_epoch(model_stack_dir, stack_size, epoch):

    files = os.listdir(model_stack_dir)

    for fname in files:

        if len(fname.split('.')) != 5:
            continue

        epoch_curr = int(fname.split('.')[1].split('-')[0])

        if epoch_curr == epoch:
            model_file_epoch = os.path.join(model_stack_dir, fname)
            val_loss = float(fname.split('.')[-3].split('-')[-1] + '.' + fname.split('.')[-2])
            return model_file_epoch, val_loss, epoch

    print('Bummer dude model_stack_dir: {}, stack_size: {},  epoch: {}'.format(
        model_stack_dir, stack_size, epoch))

    return None


def create_results_file(model_file, stack_size, cache):

    import create_results_file

    args = argparse.Namespace(
        model_file=model_file,
        stack_size=stack_size,
        input_dir=data_dir,
        output_dir=subdata_results_dir)

    create_results_file.main(args, cache)


def train_model(model_file, history_file, stack_size):
    # train_command = f'{train_file} {data_dir} odom {model_file} {history_file} -m high -e {epochs} -b 100 -s {stack_size}'
    # print(f'Training stack size {stack_size} with command: {train_command}')
    train_command = '{} {} odom {} {} -m high -e {} -b 100 -s {}'.format(train_file, data_dir, model_file, history_file, epochs, stack_size)
    # train_command = '{} {} odom {} {} -r -m high -e {} -b 100 -s {}'.format(train_file, data_dir, preload_model, history_file, epochs, stack_size)
    print('Training stack size {} with command: {}'.format(stack_size, train_command))
    subprocess.check_call(train_command,
                          shell=True,
                          env={**os.environ, 'PYTHONPATH': os.path.join(kitti_dir, 'kitti_vo')},
                          )
                          # stdout=open(os.devnull, 'wb'),
                          # stderr=open(os.devnull, 'wb'))


def eval_results():
    eval_results_cmd = '{} {}'.format(eval_bin, subdata_dir)
    # print('Running command {}'.format(eval_results_cmd))
    subprocess.check_call(eval_results_cmd,
                          shell=True,
                          env={**os.environ, 'ODO_RES_DIR': odo_res_dir, 'ODO_GT_DIR': odo_gt_dir},
                          stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

    trans_err, rot_err = open(stats_file, 'r').read().split()
    return float(trans_err), float(rot_err)

results = []

for stack_size in stack_sizes:

#    cache = {}
    cache=None   
    model_stack_dir = os.path.join(model_dir, str(stack_size))

#    shutil.rmtree(model_stack_dir)
#    os.mkdir(model_stack_dir)

    model_file = os.path.join(model_stack_dir, 'model_odom.h5')
    history_file = os.path.join(results_dir, str(stack_size), 'history.json')
    train_model(model_file, history_file, stack_size)

    models_losses = [get_model_file_epoch(model_stack_dir, stack_size, epoch) for epoch in range(1, epochs+1)]
    models_losses = [m for m in models_losses if m is not None]
    models_losses.sort(key=lambda x: x[1])

    for model_file_epoch, val_loss, epoch in models_losses[:epochs_save]:
        create_results_file(model_file_epoch, stack_size, cache)
        trans_err, rot_err = eval_results()
        result_tup = (trans_err, rot_err, stack_size, epoch, model_file_epoch, val_loss)
        results.append(result_tup)

        print('model_file_epoch: {}, val_loss: {}, epoch: {}, trans_err: {}'.format(model_file_epoch, val_loss, epoch, trans_err))


results.sort()
result_str = json.dumps(results)
with open(output_file, 'w+') as fd:
    fd.write(result_str)

for result in results:
    print('trans_err: {}, rot_err: {}, stack_size: {}, epoch: {}, val_loss: {}'.format(
        result[0], result[1], result[2], result[3], result[5]))




