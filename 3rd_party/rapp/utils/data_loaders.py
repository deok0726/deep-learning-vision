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
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.flatten())
            ]),
        )
    elif data_config['from'] in ['kaggle', 'download']:
        from sklearn.preprocessing import StandardScaler
        dset_manager = TabularDatasetManager(
            dataset_name=config.data,
            data_config=data_config,
            transform=StandardScaler(),
        )

    # balance ratio of loaders
    if use_full_class:
        seen_index_list = dset_manager.get_indexes(labels=seen_labels, ratios=[0.6, 0.2, 0.2])
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


class ImageDatasetManager:

    def __init__(
        self, dataset_name,
        transform=transforms.ToTensor(), target_transform=None,
        test_transform=None, test_target_transform=None,
        shuffle=True, data_size=0, download=True
    ):
        data_list = []
        targets_list = []

        self.train_dataset = self._get_dataset(
            dataset_name, is_train=True, download=download,
        )
        if self.train_dataset:
            data, targets = self.train_dataset.data, self.train_dataset.targets
            data_list.append(data)
            targets_list.append(targets)

        self.test_dataset = self._get_dataset(
            dataset_name, is_train=False, download=download,
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

    def _get_dataset(self, dataset_name, is_train, download, root=os.path.join(os.path.expanduser("~"), "data")):
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
                ),
                DataLoader(
                    self.test_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.test_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[2]),
                    pin_memory=use_gpu,
                    num_workers=0,
                )
            ]
