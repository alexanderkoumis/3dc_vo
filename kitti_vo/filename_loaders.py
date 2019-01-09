import math
import os
from os.path import join

import numpy as np

import stamp_parser

YAW = True


def poses_to_offsets(stamps, poses, stack_size):

    """Only to be used with load_filenames_odom"""

    def yaw_from_matrix(M):
        cy = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
        yaw = math.atan2(-M[2, 0],  cy)
        return yaw

    offsets = []

    for i in range(len(stamps)-stack_size+1):

        if i >= len(stamps)-stack_size+1 or i >= len(poses)-stack_size+1:
            break

        first_stamp, last_stamp = stamps[i], stamps[i+stack_size-1]
        first_pose, last_pose = poses[i], poses[i+stack_size-1]
        time_elapsed = last_stamp - first_stamp

        R_first, R_last = first_pose[:3, :3], last_pose[:3, :3]
        t_first, t_last = first_pose[:3, 3], last_pose[:3, 3]

        R_diff = R_last.T.dot(R_first)
        t_diff = R_first.T.dot(t_last - t_first)
        x_diff, z_diff, y_diff = t_diff

        yaw_diff = yaw_from_matrix(R_diff.T)

        # offset = np.array([y_diff, x_diff, yaw_diff])
        # offset = np.array([y_diff, yaw_diff])
        # offset = np.array([yaw_diff])
        # offset = np.array([y_diff])

        offset = np.array([yaw_diff if YAW else y_diff])

        offsets.append(offset)

    return offsets


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
    offsets_all = []

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
        offsets = poses_to_offsets(stamps, poses, stack_size)

        assert len(image_paths) == len(stamps) == len(poses) == len(offsets)+stack_size-1, '{} {} {} {}'.format(
            len(image_paths), len(stamps), len(poses), len(offsets))

        image_paths_all.append(image_paths[:-stack_size+1])
        stamps_all.append(stamps[:-stack_size+1])
        offsets_all.append(offsets)

    return image_paths_all, stamps_all, offsets_all
