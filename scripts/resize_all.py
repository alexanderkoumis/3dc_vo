#!/usr/bin/env python3.6

# import os
# import subprocess
# from fnmatch import fnmatch

# root = '/home/ubuntu/Development/kitti_vo/datasets/raw/160_90'


# def process_image(fname, rows=90, cols=160):
#     cmd = 'convert {} -resize {}x{}\\! {}'.format(fname, cols, rows, fname)
#     subprocess.call(cmd, shell=True, stdout=open(os.devnull, 'wb'))

# for path, subdirs, files in os.walk(root):
#     fnames = [fname for fname in files if fnmatch(fname, '*.png')]
#     for idx, fname in enumerate(fnames):
#         fname_full = os.path.join(path, fname)
#         print('[{}/{}] Processing {}'.format(idx+1, len(fnames), fname_full))
#         process_image(fname_full)




import os
import subprocess
from fnmatch import fnmatch

root = '/media/cache/koumis/kitti/odom/160_30/sequences'


def process_image(fname, rows=30, cols=160):
    cmd = f'convert {fname} -gravity north -extent {cols}x{rows} {fname}'
    subprocess.call(cmd, shell=True, stdout=open(os.devnull, 'wb'))

for path, subdirs, files in os.walk(root):
    fnames = [fname for fname in files if fnmatch(fname, '*.png')]
    for idx, fname in enumerate(fnames):
        fname_full = os.path.join(path, fname)
        print('[{}/{}] Processing {}'.format(idx+1, len(fnames), fname_full))
        process_image(fname_full)
