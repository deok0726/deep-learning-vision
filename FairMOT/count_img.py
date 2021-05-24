import os
import os.path as osp
import sys

pwd = "/hd/MOT/DETRAC/traindata/"
train_folders = [s for s in os.listdir(pwd)]
train_folders.sort()
count = 0

for folder in train_folders:
    train_folder = osp.join(pwd, folder)
    train_images = [s for s in os.listdir(train_folder)]
    train_images.sort()

    for image in train_images:
        count += 1

print(count)
