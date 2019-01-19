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
import cv2
from fnmatch import fnmatch

root = '/media/cache/koumis/kitti/odom/238_72/sequences'

shapes = set()

def process_image(fname, rows=72, cols=238):
	img = cv2.imread(fname)
	shapes.add(img.shape)
	return
	if img.shape[0] == rows:
		return
	img = cv2.resize(img, (cols, rows))
	cv2.imwrite(fname, img)
    # cmd = f'convert {fname} -gravity north -extent {cols}x{rows} {fname}'
    # subprocess.call(cmd, shell=True, stdout=open(os.devnull, 'wb'))

fnames_all = []

for path, subdirs, files in os.walk(root):
    fnames = [fname for fname in files if fnmatch(fname, '*.png')]
    for idx, fname in enumerate(fnames):
        fname_full = os.path.join(path, fname)
        fnames_all.append(fname_full)

for idx, fname in enumerate(fnames_all):
	print('[{}/{}] Processing {}'.format(idx+1, len(fnames_all), fname))
	process_image(fname)

print(shapes)
