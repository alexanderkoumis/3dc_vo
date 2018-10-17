
import functools
import os

import cv2
import keras.backend as K
import numpy as np


# Constants
sequences_train = ['00', '02', '08', '09']
sequences_val = ['03', '04', '05', '06', '07', '10']
sequences_test = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

stack_size = 5
yaw = 'yaw'
y = 'y'


def weighted_mse(y_true, y_pred):
    """Weighted mean squared error loss function.
    Double the loss for any yaw angle above 0.1 radians.

    Args:
        y_true (Keras tensor): Ground truth value
        y_pred (Keras tensor): Predicted value

    Returns:
        Keras tensor
    """
    high_angle = 0.1
    mask_gt = K.cast(K.abs(y_true) > high_angle, np.float32) * np.array([2.0])
    mask_lt = K.cast(K.abs(y_true) < high_angle, np.float32) * np.array([1.0])
    return K.mean(K.square(y_true - y_pred) * mask_gt + K.square(y_true - y_pred) * mask_lt)


def get_input_shape(image_paths):
    """Get input shape used for building model from image paths.
    Load in one image to find the number of rows, cols, and channels.

    Args:
        image_paths (list(list(str))): Image path stacks

    Returns:
        tuple(int, int, int, int): Model shape
    """
    rows, cols, channels = cv2.imread(image_paths[0][0]).shape
    return rows, cols, stack_size, channels


@functools.lru_cache(maxsize=100000)
def load_normalized_image(image_path, reproduce=False):
    """Loads image into memory and normalizes it.

    If reproduction argument is set, perform incorrect normalization
    procedure that was used to create paper results. It was incorrectly
    assumed that np.array() stored channel data contiguously, so calling
    flatten() would result in a 1D array like this:

        [b b b b b ... g g g g g ... r r r r r]

    However, the actual flattened layout is as follows:

        [b g r b g ... b g r b g ... b g r b g]

    Followup tests were performed using the correct normalization procedure
    (subtract mean R pixel from R channel and scale, etc.), and they showed
    results not changing by a meaningful amount.

    Args:
        image_path (str): Image
        reproduce (bool): Reproduction flag

    Returns:
        np.array((rows, cols, channels))
    """
    image = cv2.imread(image_path).astype(np.float32)

    if reproduce:

        means = [91.36508045, 74.00891816, 58.08618981]
        scales = [83.52623149, 78.35937967, 71.75622999]

        rows, cols = image.shape[:2]
        image_flat = image.flatten()
        for channel, (mean, scale) in enumerate(zip(means, scales)):
            start, end = channel * rows * cols, (channel+1) * rows * cols
            image_flat[start:end] -= mean
            image_flat[start:end] /= scale
        image = image_flat.reshape(image.shape)

    else:

        means = [104.04682693, 106.58145198, 102.51820596]
        scales = [81.44933781, 81.29074399, 80.90451186]

        image -= means
        image /= scales

    return image


def load_image_stacks(image_path_stacks, reproduce=False):
    """Loads image path stacks into memory

    Args:
        image_path_stacks (list(list(str))): List of list of image paths
        reproduce (bool): Reproduction flag

    Returns:
        np.array((num_stacks, rows, cols, stack_size, channels))
    """
    num_stacks = len(image_path_stacks)
    rows, cols, stack_size, channels = get_input_shape(image_path_stacks)

    stacks_shape = (num_stacks, rows, cols, stack_size, channels)
    image_stacks = np.zeros(stacks_shape, dtype=np.float32)

    for idx, path_stack in enumerate(image_path_stacks):
        stack_shape = (rows, cols, stack_size, channels)
        image_stack = np.zeros(stack_shape, dtype=np.float32)
        for stack_idx, path in enumerate(path_stack):
            image = load_normalized_image(path, reproduce)
            image_stack[:, :, stack_idx, 0] = image[:, :, 0]
            image_stack[:, :, stack_idx, 1] = image[:, :, 1]
            image_stack[:, :, stack_idx, 2] = image[:, :, 2]
        image_stacks[idx] = image_stack

    return image_stacks


def split_image_channels(image_stacks):
    """Converts the image data from this shape:
    
        (num_stacks, rows, cols, stack_size, channels (3))

    to this shape, which is a stack for every channel, the expected shape for the model:

       [ (num_stacks, rows, cols, stack_size, channels (1)),
         (num_stacks, rows, cols, stack_size, channels (1)),
         (num_stacks, rows, cols, stack_size, channels (1)) ]

    Args:
        image_stacks: Image stacks in one object

    Returns:
        image_stacks: Image stacks in three objects, one for each channel
    """
    image_stacks = [
        np.expand_dims(image_stacks[:, :, :, :, 0], axis=4),
        np.expand_dims(image_stacks[:, :, :, :, 1], axis=4),
        np.expand_dims(image_stacks[:, :, :, :, 2], axis=4)
    ]
    return image_stacks


def stack_data(image_paths, stamps, poses, stack_size):
    """Stacks image paths, stamps, and poses

    Args:
        image_paths: [[sequence 0 image paths], [sequence 1 image_paths], etc]
        stamps: [[sequence 0 stamps], [sequence 1 stamps], etc]
        poses: [[sequence 0 poses], [sequence 1 poses], etc]

    Returns:
        image_paths/stamps/poses: [[stack 0], [stack 1], etc]
    """
    image_paths_stacks = []
    stamps_new = []
    poses_new = []

    for image_paths_seq, stamps_seq, pose_seq in zip(image_paths, stamps, poses):

        for i in range(len(image_paths_seq)-stack_size+1):

            paths_stack = [image_paths_seq[i+j] for j in range(stack_size)]
            stamp_stack = [stamps_seq[i+j] for j in range(stack_size)]

            # If there are no poses it means we are loading from the test set
            # with no ground truth
            if len(pose_seq) > 0:
                pose_stack = [pose_seq[i+j] for j in range(stack_size)]
            else:
                pose_stack = []

            image_paths_stacks.append(paths_stack)
            stamps_new.append(stamp_stack)
            poses_new.append(pose_stack)

    return image_paths_stacks, stamps_new, poses_new



def load_filenames_odom(base_dir, sequence_nums):
    """Given the KITTI dataset base directory and list of sequences,
    load all the filenames 

    Args:
        base_dir (str): KITTI dataset directory
        sequence_nums (list(str)): List of sequence numbers

    Returns:
        tuple(list(list(str)), list(list(float)), list(list(np.array)))
    """

    def load_sequence_paths(image_dir):
        """Load the paths for a specific sequence given the directory"""
        image_paths = [fname for fname in os.listdir(image_dir)
                       if '.png' in fname and fname[0].isdigit()]
        image_paths.sort(key=lambda x: int(x.split('.')[0]))
        image_paths = [os.path.join(image_dir, fname) for fname in image_paths]
        return image_paths

    def get_stamps(stamps_path):
        """Load the stamps for a specific sequence given the filename"""
        result = []
        with open(stamps_path, 'r') as fd:
            for line in fd.readlines():
                result.append(float(line))
        return result

    def get_poses(poses_path):
        """Load the poses for a specific sequence given the filename"""
        result = []
        with open(poses_path, 'r') as fd:
            for line in fd.readlines():
                # Load the data from a string and turn it into a 4x4
                # homogeneous matrix
                pose = np.fromstring(line, dtype=float, sep=' ')
                pose = pose.reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                result.append(pose)
        return result

    image_paths_all = []
    stamps_all = []
    poses_all = []

    pose_dir = os.path.join(base_dir, 'poses')
    sequences_dir = os.path.join(base_dir, 'sequences')

    for sequence_num in sorted(sequence_nums, key=lambda x: int(x)):

        image_dir = os.path.join(sequences_dir, sequence_num, 'image_2')
        stamps_path = os.path.join(sequences_dir, sequence_num, 'times.txt')
        pose_path = os.path.join(pose_dir, '{}.txt'.format(sequence_num))

        image_paths = load_sequence_paths(image_dir)
        stamps = get_stamps(stamps_path)

        # Do not load poses for sequence numbers greater than 10, these are
        # test sequences with no associated ground truth poses
        if int(sequence_num) > 10:
            poses = []
        else:
            poses = get_poses(pose_path)

        image_paths_all.append(image_paths)
        stamps_all.append(stamps)
        poses_all.append(poses)

    return image_paths_all, stamps_all, poses_all
