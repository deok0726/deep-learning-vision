import os
# import tarfile
from PIL import Image, ImageFile
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
# CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#                'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
CLASS_NAMES = ['orange', 'gray']
# CLASS_NAMES = ['gray']

class MVTecDataset_nomask(Dataset):
    def __init__(self, data, class_name='orange', is_train=True,
                 resize=256, cropsize=224):
        # original cropsize=224
        print('daejoo_one dataset called')
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x = data
        # print(self.x)
        # print(self.y)
        # print(len(self.x))
        # print(len(self.y))

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                       T.CenterCrop(cropsize),
                                       T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])
        
        # self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
        #                                 T.CenterCrop(cropsize),
        #                                 T.ToTensor()])
        
        # self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
        #                         T.CenterCrop(cropsize),
        #                         T.ToTensor(),
        #                         T.Normalize(mean=[0.6207, 0.3677, 0.1494],    #daejoo mean
        #                                     std=[0.1571, 0.0906, 0.0365])])   #daejoo std

        # self.transform_x = T.Compose([T.ToTensor(),
        #                              T.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        print('x was called in __getitem__')
        x = self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # if y == 0:
        #     mask = torch.zeros([1, self.cropsize, self.cropsize])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return x

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                # mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                # gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                #                  for img_fname in img_fname_list]
                # mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)
