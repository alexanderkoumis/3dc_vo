import os
from os.path import join
from dateutil.parser import parser

import numpy as np


def load_filenames_raw(base_dir, stack_size, odom_idxs=[8, 9, 22]):
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
        8  vf:    forward velocity, i.e. parallel to earth-surface (m/s)
        9  vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
        10 vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
        14 af:    forward acceleration (m/s^2)
        15 al:    leftward acceleration (m/s^2)
        16 au:    upward acceleration (m/s^2)
        20 wf:    angular rate around forward axis (rad/s)
        21 wl:    angular rate around leftward axis (rad/s)
        22 wu:    angular rate around upward axis (rad/s)
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
    stamp_parser = parser()

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
        stamps = [stamp_parser.parse(val).timestamp() for val in open(oxts_stamps_txt).readlines()]

        image_paths_all.append(image_full_fnames)
        odom_all.append(odom_data)
        stamps_all.append(stamps)

    return image_paths_all, stamps_all, odom_all, num_outputs


def load_filenames_odom(base_dir, stack_size):

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

    def calc_velocities(stamps, poses):

        velocities = []

        for i in range(len(stamps)-stack_size+1):
            first_stamp, last_stamp = stamps[i], stamps[i+stack_size-1]
            first_pose, last_pose = poses[i], poses[i+stack_size-1]
            time_elapsed = last_stamp - first_stamp

            transform_world = np.linalg.inv(first_pose).dot(last_pose)
            R_world, t_world = transform_world[:3, :3], transform_world[:3, 3]
            t_cam = -R_world.T.dot(t_world)

            # t_world = last_pose[:3, 3] - first_pose[:3, 3]
            # t_cam = last_pose[:3, :3].T.dot(t_world)

            velocity = (t_cam / time_elapsed)[:2].ravel()
            velocities.append(velocity)

        return velocities

    image_paths_all = []
    stamps_all = []
    velocity_all = []
    num_outputs = 2

    pose_dir = join(base_dir, 'poses')
    sequences_dir = join(base_dir, 'sequences')

    sequence_nums = os.listdir(sequences_dir)
    for sequence_num in sequence_nums:

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
        velocities = calc_velocities(stamps, poses)

        assert len(image_paths) == len(stamps) == len(poses) == len(velocities)+stack_size-1, '{} {} {} {}'.format(
            len(image_paths), len(stamps), len(poses), len(velocities))

        image_paths_all.append(image_paths[:-stack_size+1])
        stamps_all.append(stamps[:-stack_size+1])
        velocity_all.append(velocities)

    return image_paths_all, stamps_all, velocity_all, num_outputs
