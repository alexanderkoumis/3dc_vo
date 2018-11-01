#!/usr/bin/env python3
import os
import subprocess
import shutil

import cv2


raw_base_dir = '/media/ubuntu/04EA9984EA9972A2/kitti'
processed_base_dir = '/media/ubuntu/04EA9984EA9972A2/kitti_processed'

zip_files = os.listdir(raw_base_dir)
for idx, zip_file in enumerate(zip_files):

    print('[{}/{}] Processing {}'.format(idx+1, len(zip_files), zip_file))

    zip_file_full = os.path.join(raw_base_dir, zip_file)

    subprocess.call('unzip {}'.format(zip_file_full), cwd=raw_base_dir, shell=True, stdout=open(os.devnull, 'wb'))

    date_folder = os.path.join(raw_base_dir, '_'.join(zip_file.split('_')[:3]))

    time_folder = os.listdir(date_folder)[0]
    time_folder_raw = os.path.join(date_folder, time_folder)
    time_folder_proc = os.path.join(processed_base_dir, time_folder)

    os.makedirs(time_folder_proc)

    image_dir_raw = os.path.join(time_folder_raw, 'image_02')
    oxts_dir_raw = os.path.join(time_folder_raw, 'oxts')
    image_dir_raw_data = os.path.join(time_folder_proc, 'image_02', 'data')

    shutil.copytree(image_dir_raw, os.path.join(time_folder_proc, 'image_02'))
    shutil.copytree(oxts_dir_raw, os.path.join(time_folder_proc, 'oxts'))

    image_filenames = [os.path.join(image_dir_raw_data, fname) for fname in os.listdir(image_dir_raw_data)]

    for fname in image_filenames:

        # 376 x 1241 -> 360 x 640
        img = cv2.imread(fname)[16:, 300:940]

        # 360 x 640 -> 180 x 320
        # 360 x 640 -> 180 x 360 (OOPS)
        img = cv2.resize(img, (360, 180))

        cv2.imwrite(fname, img)

    shutil.rmtree(date_folder)
    os.remove(zip_file_full)

