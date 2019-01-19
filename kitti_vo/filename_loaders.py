import math
import os
from os.path import join

import numpy as np

import stamp_parser

YAW = True


def poses_to_offsets(pose_stack, info_requested):

    def yaw_from_matrix(M):
        cy = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
        yaw = math.atan2(-M[2, 0],  cy)
        return yaw

    first_pose, last_pose = pose_stack[0], pose_stack[-1]

    R_first, R_last = first_pose[:3, :3], last_pose[:3, :3]
    t_first, t_last = first_pose[:3, 3], last_pose[:3, 3]

    R_diff = R_last.T.dot(R_first)
    t_diff = R_first.T.dot(t_last - t_first)
    x_diff, z_diff, y_diff = t_diff

    yaw_diff = yaw_from_matrix(R_diff.T)

    info = {
        'x': x_diff,
        'y': y_diff,
        'z': z_diff,
        'yaw': yaw_diff
    }

    result = np.array([info[val] for val in info_requested])

    return result


def load_filenames_odom(base_dir, stack_size, sequences=None):

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

    image_paths_all = []
    stamps_all = []
    poses_all = []

    pose_dir = join(base_dir, 'poses')
    sequences_dir = join(base_dir, 'sequences')
    sequence_nums = os.listdir(sequences_dir) if sequences is None else sequences
    sequence_nums = [num for num in sequence_nums if num.isdigit()]

    sequence_nums.sort(key=lambda x: int(x))

    for sequence_num in sequence_nums:

        if '.DS_Store' in sequence_num:
            continue

        # Only sequences 0-10 are provided for ground truth
        if int(sequence_num) > 10:
            continue

        image_dir = join(sequences_dir, sequence_num, 'image_2')
        stamps_path = join(sequences_dir, sequence_num, 'times.txt')
        pose_path = join(pose_dir, '{}.txt'.format(sequence_num))

        image_filenames = [fname for fname in os.listdir(image_dir) if '.png' in fname and fname[0].isdigit()]
        image_filenames.sort(key=lambda x: int(x.split('.')[0]))

        image_paths = [join(image_dir, fname) for fname in image_filenames]
        stamps = get_stamps(stamps_path)
        poses = get_poses(pose_path)

        image_paths_all.append(image_paths)
        stamps_all.append(stamps)
        poses_all.append(poses)

    return image_paths_all, stamps_all, poses_all
