#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import torchvision.transforms as transforms

import os
import torch
import numpy as np
from collections.abc import Iterable
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
import cv2
# import PIL
# from PIL import Image
import itertools


def get_input_size(config):
    import json
    with open('datasets/data_config.json', 'r') as f:
        data_config = json.load(f)

    data_config = data_config.get(config.data, None)
    if data_config is None:
        raise NotImplementedError
    else:
        return data_config['input_size']

def get_class_list(config):
    import json
    with open('datasets/data_config.json', 'r') as f:
        data_config = json.load(f)

    data_config = data_config.get(config.data, None)
    if data_config is None:
        raise NotImplementedError
    else:
        return data_config['labels']

def get_balance(seen_index_list, unseen_index_list, novelty_ratio=.5):
    if novelty_ratio <= 0.:
        return seen_index_list, unseen_index_list

    current_ratio = len(unseen_index_list) / (len(seen_index_list) + len(unseen_index_list))

    if current_ratio < novelty_ratio:
        target_seen_cnt = int(len(unseen_index_list) / novelty_ratio - len(unseen_index_list))
        new_seen_index_list = list(np.random.choice(seen_index_list, target_seen_cnt, replace=False))

        return new_seen_index_list, unseen_index_list
    elif current_ratio > novelty_ratio:
        target_unseen_cnt = int((len(seen_index_list) * novelty_ratio) / (1 - novelty_ratio))
        new_unseen_index_list = list(np.random.choice(unseen_index_list, target_unseen_cnt, replace=False))

        return seen_index_list, new_unseen_index_list
    else:
        return seen_index_list, unseen_index_list

def get_loaders(config, use_full_class=False):
    def random_crop(image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = image[y: y + crop_height, x: x + crop_width]
        return crop
    def random_rotation(image):
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2),np.random.randint(0, 360), np.random.rand())
        dst = cv2.warpAffine(image, M,(cols, rows))
        return dst

    # get data config for Tabular dataset
    import json
    with open('datasets/data_config.json', 'r') as f:
        data_config = json.load(f)
    if config.data not in data_config.keys():
        raise ValueError(f'no dataset config for {config.data}')
    data_config = data_config[config.data]

    # split labels 
    class_list = data_config['labels']
    seen_labels, unseen_labels = [], []

    if config.target_class not in class_list:
        config.target_class = class_list[0]
    
    for i in class_list:
        if use_full_class:
            seen_labels += [i]
            continue
        if i != config.target_class:
            if config.unimodal_normal:
                unseen_labels += [i]
            else:
                seen_labels += [i]
        else:
            if config.unimodal_normal:
                seen_labels += [i]
            else:
                unseen_labels += [i]

    if data_config['from'] == 'torchvision':
        dset_manager = ImageDatasetManager(
            dataset_name=config.data,
            transform=transforms.Compose([
                # transforms.Lambda(lambda x: random_crop(x, data_config['input_size'][1], data_config['input_size'][2])),
                # transforms.RandomCrop((data_config['input_size'][1], data_config['input_size'][2])),
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                # transforms.Lambda(lambda x: x.flatten())
            ]),
        )
    elif data_config['from'] in ['kaggle', 'download']:
        from sklearn.preprocessing import StandardScaler
        dset_manager = TabularDatasetManager(
            dataset_name=config.data,
            data_config=data_config,
            transform=StandardScaler(),
        )
    elif data_config['from'] == 'custom':
        print(data_config['input_size'][1:])
        print(tuple(data_config['input_size'][1:]))
        # def random_crop(image, crop_height, crop_width):
        #     max_x = image.shape[1] - crop_width
        #     max_y = image.shape[0] - crop_height
        #     x = np.random.randint(0, max_x)
        #     y = np.random.randint(0, max_y)
        #     crop = image[y: y + crop_height, x: x + crop_width]
        #     return crop

        dset_manager = ImageDatasetManager(
            dataset_name=config.data,
            transform=transforms.Compose([
                # transforms.Resize(data_config['input_size'][1:]),
                # transforms.Lambda(lambda x: cv2.resize(x, dsize=tuple(data_config['input_size'][1:]), interpolation=cv2.INTER_CUBIC)),
                transforms.Lambda(lambda x: random_crop(x, data_config['input_size'][1], data_config['input_size'][2])),
                transforms.Lambda(lambda x: random_rotation(x)),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: x.flatten())
            ]), data_config=data_config
        )

    # balance ratio of loaders
    if use_full_class:
        seen_index_list = dset_manager.get_indexes(labels=seen_labels, ratios=[0.66, 0.16, 0.17])
        indexes_list = [
            seen_index_list[0],
            seen_index_list[1],
            seen_index_list[2]
        ]
    else:
        seen_index_list = dset_manager.get_indexes(labels=seen_labels, ratios=[0.6, 0.2, 0.2])
        unseen_index_list = dset_manager.get_indexes(labels=unseen_labels)

        if config.verbose >= 2:
            print(
                'Before balancing:\t|train|=%d |valid|=%d |test_normal|=%d |test_novelty|=%d |novelty_ratio|=%.4f' % (
                    len(seen_index_list[0]),
                    len(seen_index_list[1]),
                    len(seen_index_list[2]),
                    len(unseen_index_list[0]),
                    len(unseen_index_list[0])/(len(unseen_index_list[0])+len(seen_index_list[2]))
                )
            )
        seen_index_list[2], unseen_index_list[0] = get_balance(seen_index_list[2],
                                                               unseen_index_list[0],
                                                               config.novelty_ratio
                                                               )
        if config.verbose >= 1:
            print(
                'After balancing:\t|train|=%d |valid|=%d |test_normal|=%d |test_novelty|=%d |novelty_ratio|=%.4f' % (
                    len(seen_index_list[0]),
                    len(seen_index_list[1]),
                    len(seen_index_list[2]),
                    len(unseen_index_list[0]),
                    len(unseen_index_list[0])/(len(unseen_index_list[0])+len(seen_index_list[2]))
                )
            )

        indexes_list = [
            seen_index_list[0],
            seen_index_list[1],
            seen_index_list[2] + unseen_index_list[0]
        ]

    train_loader, valid_loader, test_loader = dset_manager.get_loaders(
        batch_size=config.batch_size, indexes_list=indexes_list, ratios=None
    )

    return dset_manager, train_loader, valid_loader, test_loader


class SequentialIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ConcatWindowDataset(Dataset):
    def __init__(self, file_dir, target_class, window_size=1):
        self.file_list = os.listdir(file_dir)
        self.data = []        
        for name in self.file_list:
            temp = np.genfromtxt(file_dir+'/'+name)
            for i in range(len(temp)-window_size):
                self.data += [temp[i:i+window_size].reshape(-1)]
        self.data = np.array(self.data)
        self.targets = np.array([target_class] * len(self.data)).reshape(-1,1)
    
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):        
        return torch.Tensor(self.data[idx]), torch.Tensor(self.targets[idx])


class TabularDataset(Dataset):
    def __init__(self, file_dir, skip_header=0, transform=None, delimiter=None, target_transform=None):
        data = np.genfromtxt(file_dir, skip_header=skip_header, delimiter=delimiter)
        self.transform = transform
        self.target_transform = target_transform
        self.data = data[:,:-1]
        self.targets = data[:,-1]
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        x = self.data[idx].reshape(1, -1)
        y = self.targets[idx].reshape(1, -1)
        if self.transform is not None:
            x = self.transform.transform(x)
        if self.target_transform is not None:
            y = self.target_transform.transform(y)
        
        return torch.Tensor(x.squeeze()), torch.Tensor(y.squeeze())


class TabularDatasetManager:
    
    def __init__(self, dataset_name, data_config,
        transform=None, target_transform=None,
        test_transform=None, test_target_transform=None,
        shuffle=True, data_size=0):

        data_list = []
        targets_list = []

        self.train_dataset = self._get_dataset(
            dataset_name, is_train=True, data_config=data_config
        )
        if self.train_dataset:
            data, targets = self.train_dataset.data, self.train_dataset.targets
            data_list.append(data)
            targets_list.append(targets)

        if type(data[0]) == np.ndarray:
            self.total_x = np.concatenate(data_list)[-data_size:] if data_list else None
            self.total_y = np.concatenate(targets_list)[-data_size:] if targets_list else None
        elif type(data[0]) == torch.Tensor:
            self.total_x = torch.cat(data_list)[-data_size:] if data_list else None
            self.total_y = torch.cat(targets_list)[-data_size:] if targets_list else None
        else:
            raise NotImplementedError

        if shuffle:
            from numpy.random import permutation
            shuffled_indices = permutation(len(self.total_x))
            self.total_x = self.total_x[shuffled_indices]
            self.total_y = self.total_y[shuffled_indices]

        self.total_size = len(self.total_x)

        self.train_dataset.data = self.total_x
        self.train_dataset.targets = self.total_y
        self.train_dataset.transform = transform
        self.train_dataset.target_transform = target_transform
        
    def _get_dataset(
        self,
        dataset_name,
        is_train,
        data_config,
        root=os.path.join(os.path.expanduser("~"), "data"),
    ):

        from datasets.data_preprocess import get_preprocess, get_data_from_url
        dataset_name = dataset_name.lower()
        # process with kaggle data
        if data_config['from'] == 'kaggle':
            folder_name = data_config['folder_name'].replace('/', '')
            if folder_name not in os.listdir(root):
                print(f'{dataset_name} does not exist on root data folder, start to get data')
                get_preprocess(dataset_name, data_config, root)
                print(f'{dataset_name} are ready')
            elif data_config['file_name'] not in os.listdir(os.path.join(root, folder_name, 'processed')):
                print(f'{dataset_name} file does not exist on {folder_name}, start to get data')
                get_preprocess(dataset_name, data_config, root)
                print(f'{dataset_name} are ready')
            load_dir = os.path.join(root, data_config['folder_name'], 'processed', data_config['file_name'])
            dataset = TabularDataset(load_dir, skip_header=1, delimiter=',')    
        #process with download data
        elif data_config['from'] == 'download':
            if data_config['file_name'] not in os.listdir(root):
                get_data_from_url(data_config['url'], data_config['file_name'], root)
            load_dir = os.path.join(root, data_config['file_name'])
            dataset = TabularDataset(load_dir, skip_header=data_config['skip_header'], delimiter=data_config['delimiter'])
        elif dataset_name in ['wafer_1', 'wafer_2']:
            if 'wafer_1' not in os.listdir(root):
                root = '/mnt/mrx-nas/jongseob.jeon/data/'
            dataset = TabularDataset(root + '/' + dataset_name, skip_header=1, delimiter=',')
        else:
            raise NotImplementedError("No data for {}".format(dataset_name))
        return dataset

    def get_indexes(self, ratios=None, labels=None):
        if labels is not None: 
            if not isinstance(labels, Iterable):
                labels = [labels]
            indexes = list(np.where(np.isin(self.total_y, labels))[0])
        else:
            indexes = list(range(self.total_size))

        if ratios:
            assert sum(ratios) == 1
            if len(ratios) == 1:
                return indexes
            else:
                ratios = np.array(ratios)
                indexes_list = np.split(indexes, [int(e) for e in (ratios.cumsum()[:-1] * len(indexes))])
                indexes_list = [list(indexes) for indexes in indexes_list]
        else:
            indexes_list = [indexes]

        return indexes_list

    def get_transformed_data(self, data_loader):
        """
        Multi indexing support
        """
        x = []
        y = []
        for i in data_loader.sampler:
            # indexing -> __getitem__ -> applying transform
            _x, _y = data_loader.dataset[i]
            x.append(_x)
            y.append(_y)

        if type(_x) == np.ndarray:
            x = np.stack(x)
        elif type(_x) == torch.Tensor:
            x = torch.stack(x)
        else:
            raise NotImplementedError

        if type(_y) == np.ndarray:
            y = np.array(y)
        elif type(_y) == torch.Tensor:
            y = torch.tensor(y)

        return x, y
    
    def get_loaders(self, batch_size, ratios=None, indexes_list=None, use_gpu=False):
        if ratios and indexes_list:
            raise Exception("Only either `ratios` or `indexes_list` is allowed")
        elif ratios:
            indexes_list = self.get_indexes(ratios=ratios)

        if self.train_dataset.transform is not None:
            self.train_dataset.transform.fit(self.train_dataset.data[indexes_list[0]])

        if len(indexes_list) == 2:
            return [
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SubsetRandomSampler(indexes_list[0]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
            ]
        elif len(indexes_list) == 3:
            return [
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SubsetRandomSampler(indexes_list[0]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[2]),
                    pin_memory=use_gpu,
                    num_workers=0,
                )
            ]

# class ImageDataset_old(Dataset):
#     def __init__(self, root, train=True, transform=None, target_transform=None):
#         # data = np.genfromtxt(file_dir, skip_header=skip_header, delimiter=delimiter)
#         self.image_paths = []
#         self.labels = []
#         if(train):
#             root_dir = os.path.join(root, 'train')
#         else:
#             root_dir = os.path.join(root, 'test')
#         for folder in os.listdir(root_dir):
#             folder_path = os.path.join(root_dir, folder)
#             images_in_folder = sorted(os.listdir(folder_path))
#             self.image_paths.extend([os.path.join(folder_path, image) for image in images_in_folder])
#             if folder == 'good':
#                 self.labels.extend([0]*len(images_in_folder))
#             else:
#                 self.labels.extend([1]*len(images_in_folder))
#         assert len(self.image_paths) == len(self.labels), 'images and label number mismatch'
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.image_paths)
        
#     def __getitem__(self, idx):
#         x = self.image_paths[idx]
#         y = self.labels[idx]
#         img = cv2.imread(x)
#         if self.transform is not None:
#             x = self.transform.transform(x)
#         if self.target_transform is not None:
#             y = self.target_transform.transform(y)
        
#         return torch.Tensor(x), torch.Tensor(y)

class ImageDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.data = []
        self.targets = []
        transToTensor = transforms.ToTensor()
        if(train):
            root_dir = os.path.join(root, 'train')
        else:
            root_dir = os.path.join(root, 'test')
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            images_in_folder = sorted(os.listdir(folder_path))
            self.data.extend([cv2.imread(os.path.join(folder_path, image)) for image in images_in_folder])
            # self.data.extend([transToTensor(Image.open(os.path.join(folder_path, image))) for image in images_in_folder])
            # self.data.extend([Image.open(os.path.join(folder_path, image)).convert("RGB") for image in images_in_folder])
            if folder == 'good':
                self.targets.extend([0]*len(images_in_folder))
            else:
                self.targets.extend([1]*len(images_in_folder))
        assert len(self.data) == len(self.targets), 'images and label number mismatch'
        # print(type(self.data[0]))
        # print(type(self.data[0])==PIL.PngImagePlugin.PngImageFile)
        self.data = np.array(self.data)
        # self.data = torch.stack(self.data)
        self.targets = np.array(self.targets)
        # self.targets = torch.FloatTensor(self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx].reshape(1, -1)
        # y = self.targets[idx]
        # img = cv2.imread(x)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        
        return torch.Tensor(x.squeeze()), torch.Tensor(y)
        # return torch.Tensor(x), torch.Tensor(y)

class ImageDatasetManager:

    def __init__(
        self, dataset_name,
        transform=transforms.ToTensor(), target_transform=None,
        test_transform=None, test_target_transform=None,
        shuffle=True, data_size=0, download=True, data_config=None
    ):
        data_list = []
        targets_list = []

        self.train_dataset = self._get_dataset(
            dataset_name, is_train=True, download=download, data_config=data_config
        )
        if self.train_dataset:
            data, targets = self.train_dataset.data, self.train_dataset.targets
            data_list.append(data)
            targets_list.append(targets)

        self.test_dataset = self._get_dataset(
            dataset_name, is_train=False, download=download, data_config=data_config
        )
        if self.test_dataset:
            data, targets = self.test_dataset.data, self.test_dataset.targets
            data_list.append(data)
            targets_list.append(targets)

        if type(data[0]) == np.ndarray:
            self.total_x = np.concatenate(data_list)[-data_size:] if data_list else None
            self.total_y = np.concatenate(targets_list)[-data_size:] if targets_list else None
        elif type(data[0]) == torch.Tensor:
            self.total_x = torch.cat(data_list)[-data_size:] if data_list else None
            self.total_y = torch.cat(targets_list)[-data_size:] if targets_list else None
        # elif type(data[0]) == PIL.Image.Image:
            # self.total_x = list(itertools.chain.from_iterable(data_list))[-data_size:] if data_list else None
            # self.total_y = list(itertools.chain.from_iterable(targets_list))[-data_size:] if targets_list else None
        else:
            raise NotImplementedError

        if shuffle:
            from numpy.random import permutation
            shuffled_indices = permutation(len(self.total_x))
            self.total_x = self.total_x[shuffled_indices]
            self.total_y = self.total_y[shuffled_indices]

        self.total_size = len(self.total_x)

        self.train_dataset.data = self.total_x
        self.train_dataset.targets = self.total_y
        self.train_dataset.transform = transform
        self.train_dataset.target_transform = target_transform

        self.test_dataset.data = self.total_x
        self.test_dataset.targets = self.total_y
        self.test_dataset.transform = test_transform if test_transform else transform
        self.test_dataset.target_transform = test_target_transform if test_target_transform else target_transform

    def _get_dataset(self, dataset_name, is_train, download, data_config, root=os.path.join(os.path.expanduser("~"), "data")):
        dataset_name = dataset_name.lower()
        if dataset_name == "mnist":
            dataset = MNIST(
                root=root,
                train=is_train,
                download=download,
            )
        elif dataset_name == "fmnist":
            dataset = FashionMNIST(
                root=root,
                train=is_train,
                download=download,
            )
        elif dataset_name == "mvtec":
            folder_name = data_config['folder_name'].replace('/', '')
            load_dir = os.path.join(root, folder_name)
            dataset = ImageDataset(load_dir, train=is_train)
        else:
            raise NotImplementedError("No data for {}".format(dataset_name))
        return dataset

    def get_indexes(self, ratios=None, labels=None):
        if labels is not None:   
            if not isinstance(labels, Iterable):
                labels = [labels]
            indexes = list(np.where(np.isin(self.total_y, labels))[0])
        else:
            indexes = list(range(self.total_size))

        if ratios:
            assert sum(ratios) == 1
            if len(ratios) == 1:
                return indexes
            else:
                ratios = np.array(ratios)
                indexes_list = np.split(indexes, [int(e) for e in (ratios.cumsum()[:-1] * len(indexes))])
                indexes_list = [list(indexes) for indexes in indexes_list]
        else:
            indexes_list = [indexes]

        return indexes_list

    def get_transformed_data(self, data_loader):
        """
        Multi indexing support
        """
        x = []
        y = []
        for i in data_loader.sampler:
            # indexing -> __getitem__ -> applying transform
            _x, _y = data_loader.dataset[i]
            x.append(_x)
            y.append(_y)

        if type(_x) == np.ndarray:
            x = np.stack(x)
        elif type(_x) == torch.Tensor:
            x = torch.stack(x)
        else:
            raise NotImplementedError

        if type(_y) == np.ndarray:
            y = np.array(y)
        elif type(_y) == torch.Tensor:
            y = torch.tensor(y)

        return x, y

    def get_loaders(self, batch_size, ratios=None, indexes_list=None, use_gpu=False):
        if ratios and indexes_list:
            raise Exception("Only either `ratios` or `indexes_list` is allowed")
        elif ratios:
            indexes_list = self.get_indexes(ratios=ratios)

        if len(indexes_list) == 2:
            return [
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SubsetRandomSampler(indexes_list[0]),
                    pin_memory=use_gpu,
                    num_workers=0
                ),
                DataLoader(
                    self.test_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0
                ),
            ]
        elif len(indexes_list) == 3:
            return [
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SubsetRandomSampler(indexes_list[0]),
                    pin_memory=use_gpu,
                    num_workers=0,
                    drop_last=True
                ),
                DataLoader(
                    self.test_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0,
                    drop_last=True
                ),
                DataLoader(
                    self.test_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[2]),
                    pin_memory=use_gpu,
                    num_workers=0,
                    drop_last=True
                )
            ]
