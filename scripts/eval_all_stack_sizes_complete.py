#!/usr/bin/env python3.6

import json
import os
import shutil
import subprocess
import sys



stack_sizes = [3, 5, 7, 10]


kitti_dir = '/home/koumis/Development/kitti_vo'
data_dir = '/media/cache/koumis/kitti/odom/160_90'
eval_bin = '/home/koumis/Development/External/kitty_eval/evaluate_odometry_quiet'

os.environ['ODO_GT_DIR'] = os.path.join(data_dir, 'poses')
os.environ['ODO_RES_DIR'] = os.path.join(data_dir, 'poses_results')

model_dir = os.path.join(kitti_dir, 'models', 'odom')
results_dir = os.path.join(kitti_dir, 'results', 'odom')

main_file = os.path.join(kitti_dir, 'kitti_vo', 'main.py')
plot_file = os.path.join(kitti_dir, 'kitti_vo', 'plot.py')

create_results_script = os.path.join(kitti_dir, 'scripts', 'create_results_file.py')

subdata_dir = 'cool'
subdata_results_dir = os.path.join(os.environ['ODO_RES_DIR'], subdata_dir, 'data')
stats_file = os.path.join(os.environ['ODO_RES_DIR'], subdata_dir, 'stats.txt')
output_file = os.path.join(os.environ['ODO_RES_DIR'], subdata_dir, 'all_stack_results.txt')

shutil.rmtree(os.environ['ODO_RES_DIR'])
os.mkdir(os.environ['ODO_RES_DIR'])
os.makedirs(subdata_results_dir)




def eval_model(model_file, stack_size, epoch):
	create_results_cmd = f'{create_results_script} {model_file} {stack_size} {data_dir} {subdata_results_dir}'
	print(f'Running command {create_results_cmd}')
	subprocess.check_call(create_results_cmd, shell=True)

	with open(stats_file, 'r') as fd:
		trans_err, rot_err = fd.read().split()
		return float(trans_err), float(rot_err)

def train_model(model_file, stack_size):
    train_command = f'{main_file} {data_dir} {model_file} odom -m high -e 200 -b 100 -s {stack_size}'
    print(f'Training stack size {stack_size} with command: {train_command}')
    subprocess.check_call(train_command, shell=True)



results = []

for stack_size in stack_sizes:

	model_file = os.path.join(model_dir, str(stack_size), 'model_odom.h5')
	train_model(model_file, stack_size)

    for epoch in range(180, 200):

    	trans_err, rot_err = eval_model(model_file, stack_size, epoch)
    	result_tup = (trans_err, rot_err, stack_size, epoch)
    	results.append(result_tup)


results.sort()

result_str = json.dumps(results)
with open(output_file, 'w+') as fd:
	fd.write(results_str)

for result in results:
	print('trans_err: {}, rot_err: {}, stack_size: {}, epoch: {}'.format(
		result[0], result[1], result[2], result[3]))





