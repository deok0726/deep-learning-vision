import os
import os.path as osp
import sys

pwd = "/hd/MOT/DETRAC/labels_with_ids/"
write_data = "/root/FairMOT/src/data/detrac.train"
train_folders = [s for s in os.listdir(pwd)]
train_folders.sort()

for folder in train_folders:
    train_folder = osp.join(pwd, folder)
    train_images = [s for s in os.listdir(train_folder)]
    train_images.sort()
    # train_images.remove('seqinfo.ini')

    for image in train_images:
        img_dir = osp.join(train_folder, image)
        img_dir = img_dir.replace('labels_with_ids', 'traindata').replace('.txt', '.jpg')
        with open(write_data, 'a') as f:
            f.write(img_dir[8:] + '\n')

f.close()