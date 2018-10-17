#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

import common


def is_float(num_str):
    """Determines if a number (string) is a float

    Args:
        num_str (str): The number string being tested

    Returns:
        bool: Whether or not num_str is a float
    """
    try:
        float(num_str)
        return True
    except:
        return False


def exec_command(command, verbose=False):
    """Runs a shell command

    Args:
        command (str): The shell command to be executed
        verbose (bool): Whether or not to print the output of the command (used for debugging)
    """
    if verbose:
        print('Executing command: {}'.format(command))
        subprocess.check_call(command, shell=True)
    else:
        subprocess.check_call(command, shell=True,
                              stdout=open(os.devnull, 'wb'),
                              stderr=open(os.devnull, 'wb'))


def create_results_file(create_results_script, input_dir, output_dir, yaw_or_y,
                        model_pretrained, model_curr):
    """Generate the results files (in the KITTI output format ) using the provided model file

    Args:
        create_results_script (str): Path to script that creates result files
        input_dir (str): Directory where images are
        output_dir (str): Directory where results are stored
        yaw_or_y (str): Are we testing y or yaw models?
        model_pretrained (str): The unchanging, pretrained model (if yaw_or_y is 'y', this should be
                                a pretrained yaw model, and vice versa)
        model_curr (str): The current model being tested (if yaw_or_y is 'y', this should be the
                          y model currently being tested)
    """
    yaw_or_y_pretrained = common.yaw if yaw_or_y == common.y else common.y

    model_arg = '--model_{} {} --model_{} {}'.format(yaw_or_y, model_curr,
                                                     yaw_or_y_pretrained, model_pretrained)

    command = '{} --input_dir {} --output_dir {} --scale_yaw {} --scale_y {} {}'.format(
        create_results_script, input_dir, output_dir, 1.0, 1.0, model_arg)

    exec_command(command)


def eval_results(eval_bin, subdata_dir, odo_gt_dir, odo_res_dir):
    """Run the official KITTI evaluation tool on the results output from
    create_results_file.py. Return the average translational and rotational
    error.

    Args:
        eval_bin (str): Path to evaluation binary
        subdata_dir (str): Name of subdirectory in results file
        odo_gt_dir (str): Directory where ground truth poses are
        odo_res_dir (str): Directory where output from create_results_file.py is

    Returns:
        tuple(float, float): Average translational and rotational error of results
    """
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


def get_models_losses(model_stack_dir, epochs_max=10000):
    """Go through directory and find all models within the specified epoch range

    Args:
        model_stack_dir (path): Model directory
        epochs_max (int): Max epoch to search for

    Returns:
        list(str): Sorted list of models
    """
    def get_model_file_epoch(model_stack_dir, epoch):
        """Find a model filename corresponding to a specific epoch"""
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

    for epoch in range(1, epochs_max+1):
        models_losses += get_model_file_epoch(model_stack_dir, epoch)

    return sorted(models_losses, key=lambda x: x[1])


def print_results(results, num_display=100):
    """Pretty print results

    Args:
        results (list(list(float, float, int, float))): Results
    """
    for result in results[:num_display]:
        print('trans_err: {}, rot_err: {}, epoch: {}, val_loss: {}'.format(
            result[0], result[1], result[2], result[4]))


def main(args):
    """For every model in the specified folder, evaluate it using
    the provided KITTI evaluation tool, sort the results, and display them.
    This helps pick the best model to use.
    """
    this_dir = os.path.dirname(os.path.realpath(__file__))
    kitti_dir = os.path.dirname(this_dir)

    create_results_script = os.path.join(this_dir, 'create_results_file.py')

    odo_gt_dir = os.path.join(args.data_dir, 'poses/')
    odo_res_dir = os.path.join(args.data_dir, 'poses_results/')

    subdata_dir = 'cool'
    results_dir = os.path.join(odo_res_dir, subdata_dir, 'data')

    results = []

    models_losses = get_models_losses(args.model_dir)

    for model_epoch, val_loss, epoch in models_losses[:args.epochs_save]:

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
    parser.add_argument('epochs_save', type=int, default=500, help='Number of models to evaluate')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
