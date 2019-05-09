#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

import common


EPOCHS_TOTAL = 3000
EPOCHS_SAVE = 500


def is_float(num_str):
    try:
        float(num_str)
        return True
    except:
        return False


def exec_command(command, verbose=False):
    if verbose:
        print('Executing command: {}'.format(command))
        subprocess.check_call(command, shell=True)
    else:
        subprocess.check_call(command, shell=True,
                              stdout=open(os.devnull, 'wb'),
                              stderr=open(os.devnull, 'wb'))


def create_results_file(create_results_script, input_dir, output_dir, yaw_or_y, model_pretrained, model_curr):

    yaw_or_y_pretrained = common.yaw if yaw_or_y == common.y else common.y

    model_arg = '--model_{} {} --model_{} {}'.format(yaw_or_y, model_curr,
                                                     yaw_or_y_pretrained, model_pretrained)

    command = '{} --input_dir {} --output_dir {} --scale_yaw {} --scale_y {} {}'.format(
        create_results_script, input_dir, output_dir, 1.0, 1.0, model_arg)

    exec_command(command)


def eval_results(eval_bin, subdata_dir, odo_gt_dir, odo_res_dir):

    plot_error_dir = os.path.join(odo_res_dir, subdata_dir, 'plot_error')
    plot_path_dir = os.path.join(odo_res_dir, subdata_dir, 'plot_path')
    errors_dir = os.path.join(odo_res_dir, subdata_dir, 'errors')

    # Clean up
    shutil.rmtree(plot_error_dir, ignore_errors=True)
    shutil.rmtree(plot_path_dir, ignore_errors=True)
    shutil.rmtree(errors_dir, ignore_errors=True)

    eval_results_cmd = '{} {} {} {}'.format(eval_bin, subdata_dir, odo_gt_dir, odo_res_dir)
    exec_command(eval_results_cmd)

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
    for sequence in common.sequences_val:
        tl_file = os.path.join(plot_error_dir, '{}_tl.txt'.format(sequence))
        rl_file = os.path.join(plot_error_dir, '{}_rl.txt'.format(sequence))
        tl_avgs.append(get_kitti_eval_avg(tl_file))
        rl_avgs.append(get_kitti_eval_avg(rl_file, 57.2958))
    tl_avg = statistics.mean(tl_avgs)
    rl_avg = statistics.mean(rl_avgs)

    return tl_avg, rl_avg


def get_models_losses(model_stack_dir, num_epochs):

    def get_model_file_epoch(model_stack_dir, epoch):

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

        return result

    models_losses = []

    for epoch in range(1, num_epochs+1):
        models_losses += get_model_file_epoch(model_stack_dir, epoch)

    models_losses.sort(key=lambda x: x[1])
    return models_losses


def print_results(r):
    for result in r[:100]:
        print('trans_err: {}, rot_err: {}, epoch: {}, val_loss: {}'.format(
            result[0], result[1], result[2], result[4]))


def main(args):

    this_dir = os.path.dirname(os.path.realpath(__file__))
    kitti_dir = os.path.dirname(this_dir)

    create_results_script = os.path.join(this_dir, 'create_results_file.py')

    odo_gt_dir = os.path.join(args.data_dir, 'poses/')
    odo_res_dir = os.path.join(args.data_dir, 'poses_results/')

    subdata_dir = 'cool'
    results_dir = os.path.join(odo_res_dir, subdata_dir, 'data')

    results = []

    models_losses = get_models_losses(args.model_dir, EPOCHS_TOTAL)

    for model_epoch, val_loss, epoch in models_losses[:EPOCHS_SAVE]:

        create_results_file(create_results_script, args.data_dir, results_dir,
                            args.yaw_or_y, args.model_pretrained, model_epoch)

        trans_err, rot_err = eval_results(args.eval_bin, subdata_dir, odo_gt_dir, odo_res_dir)
        result_tup = (trans_err, rot_err, epoch, model_epoch, val_loss)
        results.append(result_tup)
        results.sort(key=lambda x: x[0 if args.yaw_or_y == common.y else 1])

        print('val_loss: {}, epoch: {}, trans_err: {}, rot_err: {}'.format(
            val_loss, model_epoch.split('/')[-1], trans_err, rot_err))

        print_results(results)
        print('\n\n\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_bin', help='Evaluate binary filename')
    parser.add_argument('data_dir', help='Dataset directory')
    parser.add_argument('model_dir', help='Model directory')
    parser.add_argument('yaw_or_y', choices=[common.yaw, common.y], help='Evaluating yaw or y')
    parser.add_argument('model_pretrained', help='Other (pretrained) model. If yaw_or_y == yaw, '
                                                  'model_pretrained is your pretrained y model.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
