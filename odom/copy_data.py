#!/usr/bin/env python3

"""
Creates cropped/resized version of KITTI dataset,
only includes left camera

376x1241 --crop--> 360x640 --resize--> 180x320
"""

import os
import shutil

import cv2


sequences_dir_src = '/media/ubuntu/42d96c1f-4e3d-43b6-8e21-0c922afca907/kitti_rgb_original/sequences/'
sequences_dir_dst = '/media/ubuntu/42d96c1f-4e3d-43b6-8e21-0c922afca907/kitti_rgb_resized/sequences'

sequence_nums = os.listdir(sequences_dir_src)

for seq_idx, sequence_num in enumerate(sequence_nums):

    seq_dir_src = os.path.join(sequences_dir_src, sequence_num)
    seq_dir_dst = os.path.join(sequences_dir_dst, sequence_num)

    shutil.copy(os.path.join(seq_dir_src, 'calib.txt'), os.path.join(seq_dir_dst, 'calib.txt'))   
    shutil.copy(os.path.join(seq_dir_src, 'times.txt'), os.path.join(seq_dir_dst, 'times.txt'))

    image_dir_src = os.path.join(seq_dir_src, 'image_2')
    image_dir_dst = os.path.join(seq_dir_dst, 'image_2')

    os.makedirs(image_dir_dst)

    image_names = os.listdir(image_dir_src)

    for img_idx, img_name in enumerate(image_names):

        img_name_full_src = os.path.join(image_dir_src, img_name)
        img_name_full_dst = os.path.join(image_dir_dst, img_name)

        print('Processing sequence {}/{}, image {}/{}: {}'.format(
            seq_idx, len(sequence_nums), img_idx, len(image_names), img_name_full_dst))

        # 376 x 1241 -> 360 x 640
        img = cv2.imread(img_name_full_src)[16:, 300:940]

        # 360 x 640 -> 180 x 320
        img = cv2.resize(img, (360, 180))

        cv2.imwrite(img_name_full_dst, img)

