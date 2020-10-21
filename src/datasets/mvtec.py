import sys
import os
from torch.utils.data import Subset
from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from skimage import io


class MVTEC_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 15))
        self.outlier_classes.remove(normal_class)

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1'))])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
    
        self.train_set = MyMVTEC(root=self.root, train=True, transform=transform,
                            target_transform=target_transform, normal_class=normal_class)
        
        # Subset train_set to normal class
        # train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        # self.train_set = Subset(train_set, train_idx_normal)
        
        self.test_set = MyMVTEC(root=self.root, train=False, transform=transform,
                                target_transform=target_transform, normal_class=normal_class)


class MyMVTEC(Dataset):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train, transform, target_transform, normal_class):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.mvtec_part = sorted(os.listdir(root))[normal_class]
        print('Normal class is', self.mvtec_part)
        self.root = os.path.join(root, self.mvtec_part)
        self.train_dir = os.path.join(self.root, "train/good/")
        self.test_dir = os.path.join(self.root, "test/")
        self.train_data = sorted(os.listdir(self.train_dir))
        self.test_folders = sorted(os.listdir(self.test_dir))
        self.test_data = []
        self.train_target = []
        self.test_target = []
        self.train_target.extend([0]*len(self.train_data))

        for folder in self.test_folders:
            test_folders_dir = os.path.join(self.test_dir, folder)
            test_folders_data = sorted(os.listdir(test_folders_dir))
            self.test_data.extend([os.path.join(test_folders_dir, data) for data in test_folders_data])
            if folder == 'good':
                self.test_target.extend([0]*len(test_folders_data))
            else:
                self.test_target.extend([1]*len(test_folders_data))
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img = io.imread(os.path.join(self.train_dir, self.train_data[index]))
            target = self.train_target[index]
        else:
            img = io.imread(self.test_data[index])
            target = self.test_target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, index  # only line changed