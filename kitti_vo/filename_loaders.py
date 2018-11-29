import math
import os
from os.path import join

import numpy as np

import stamp_parser


def load_filenames_raw(base_dir, stack_size, odom_idxs=[8, 9, 5], sequences=None):
    """
    Directory structure:
        base_dir/
            2011_09_26_drive_0001_sync/
                image_02/
                    timestamps.txt
                    data/
                        [d+].png
                oxts/
                    dataformat.txt
                    timestamps.txt
                    data/
                        [d+].txt

    Odom data format:
        5  yaw: heading (rad), 0 = east, positive = counter clockwise, range: -pi .. +pi
        8   vf: forward velocity, i.e. parallel to earth-surface (m/s)
        9   vl: leftward velocity, i.e. parallel to earth-surface (m/s)
        10  vu: upward velocity, i.e. perpendicular to earth-surface (m/s)
        14  af: forward acceleration (m/s^2)
        15  al: leftward acceleration (m/s^2)
        16  au: upward acceleration (m/s^2)
        20  wf: angular rate around forward axis (rad/s)
        21  wl: angular rate around leftward axis (rad/s)
        22  wu: angular rate around upward axis (rad/s)
    """

    def load_odometry(odom_path, image_name):
        """Load target odometry"""
        full_path = join(odom_path, '{}.txt'.format(image_name))
        with open(full_path, 'r') as fd:
            data = fd.read().split()
            vals = [float(data[idx]) for idx in odom_idxs]
            return vals

    image_paths_all = []
    odom_all = []
    stamps_all = []
    num_outputs = len(odom_idxs)

    sequences = os.listdir(base_dir)
    parser = stamp_parser.StampParser()

    for sequence in sequences:

        sequence_dir = join(base_dir, sequence)

        image_data_dir = join(sequence_dir, 'image_02', 'data')
        oxts_dir = join(sequence_dir, 'oxts')
        oxts_data_dir = join(oxts_dir, 'data')

        oxts_stamps_txt = join(oxts_dir, 'timestamps.txt')

        image_fnames = os.listdir(image_data_dir)
        image_fnames.sort(key=lambda x: int(x.split('.')[0]))
        image_names = [path.split('.')[0] for path in image_fnames]

        image_full_fnames = [join(image_data_dir, fname) for fname in image_fnames]
        odom_data = [load_odometry(oxts_data_dir, name) for name in image_names]
        stamps = [parser.parse(val) for val in open(oxts_stamps_txt).readlines()]

        image_paths_all.append(image_full_fnames)
        odom_all.append(odom_data)
        stamps_all.append(stamps)

    return image_paths_all, stamps_all, odom_all, num_outputs


def poses_to_velocities(stamps, poses, stack_size):

    """Only to be used with load_filenames_odom"""

    def yaw_from_matrix(M):
        cy = math.sqrt(M[0, 0]*M[0, 0] + M[1, 0]*M[1, 0])
        yaw = math.atan2(-M[2, 0],  cy)
        return yaw

    velocities = []

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

        # velocity = np.array([y_diff, x_diff, yaw_diff]) / time_elapsed
        velocity = np.array([y_diff, x_diff, yaw_diff])
        velocities.append(velocity)

    return velocities


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
    velocities_all = []
    num_outputs = 3

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

        image_filenames = os.listdir(image_dir)
        image_filenames.sort(key=lambda x: int(x.split('.')[0]))

        image_paths = [join(image_dir, fname) for fname in image_filenames]
        stamps = get_stamps(stamps_path)
        poses = get_poses(pose_path)
        velocities = poses_to_velocities(stamps, poses, stack_size)

        assert len(image_paths) == len(stamps) == len(poses) == len(velocities)+stack_size-1, '{} {} {} {}'.format(
            len(image_paths), len(stamps), len(poses), len(velocities))

        image_paths_all.append(image_paths[:-stack_size+1])
        stamps_all.append(stamps[:-stack_size+1])
        velocities_all.append(velocities)

    return image_paths_all, stamps_all, velocities_all, num_outputs
