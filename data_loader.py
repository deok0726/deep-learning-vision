from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from PIL import Image
import argument_parser as parser
import torch
import os
import random
import numpy as np


class DataLoader:

    def __init__(self, args):
        source_transform_list = self._get_transform(args, is_target=False)
        target_transform_list = self._get_transform(args, is_target=True)
        train_dataset, valid_dataset, test_dataset = self._get_datasets(args, source_transform=source_transform_list, target_transform=target_transform_list)
        sample_train_data = train_dataset.__getitem__(np.random.randint(train_dataset.__len__()))[0]
        self.sample_train_data = sample_train_data.unsqueeze(0)
        self.train_data_loader = self._get_data_loader(args, train_dataset)
        self.valid_data_loader = self._get_data_loader(args, valid_dataset)
        self.test_data_loader = self._get_data_loader(args, test_dataset)

    def _get_datasets(self, args, source_transform, target_transform):
        if args.dataset_name == 'MNIST':
            train_dataset = MNIST(root=args.dataset_root, train=True, download=True, transform=source_transform, target_transform=target_transform)
            test_dataset = MNIST(root=args.dataset_root, train=False, download=True, transform=source_transform, target_transform=target_transform)
            train_dataset, valid_dataset, test_dataset = self._preprocess_to_anomaly_detection_dataset(args, train_dataset, test_dataset)
        elif args.dataset_name == 'FMNIST':
            train_dataset = FashionMNIST(root=args.dataset_root, train=True, download=True, transform=source_transform, target_transform=target_transform)
            test_dataset = FashionMNIST(root=args.dataset_root, train=False, download=True, transform=source_transform, target_transform=target_transform)
            train_dataset, valid_dataset, test_dataset = self._preprocess_to_anomaly_detection_dataset(args, train_dataset, test_dataset)
        elif args.dataset_name == 'MVTEC':
            train_dataset = ImageDataset(root=args.dataset_root, train=True, transform=source_transform, target_transform=target_transform)
            test_dataset = ImageDataset(root=args.dataset_root, train=False, transform=source_transform, target_transform=target_transform)
            train_length = int(len(train_dataset) * (args.train_ratio / (args.train_ratio + args.valid_ratio)))
            valid_length = int(len(train_dataset) * (args.valid_ratio / (args.train_ratio + args.valid_ratio)))
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_length, valid_length])
        else:
            raise Exception('Wrong dataset name')
        return train_dataset, valid_dataset, test_dataset
    
    def _preprocess_to_anomaly_detection_dataset(self, args, original_train_dataset, original_test_dataset):
        # Divide normal data according to the ratio of training set: validation set: test set
        # ================================================================== #
        #            1. Combine original training set and test set           #
        # ================================================================== #
        total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
        train_ratio = args.train_ratio / total_ratio
        valid_ratio = args.valid_ratio / total_ratio
        test_ratio = args.test_ratio / total_ratio
        train_data, train_targets = original_train_dataset.data, original_train_dataset.targets
        test_data, test_targets = original_test_dataset.data, original_test_dataset.targets
        data = torch.cat((train_data, test_data))
        targets = torch.cat((train_targets, test_targets))
        if isinstance(data, torch.ByteTensor):
            data = data.float().div(255)
        # ============================================================================= #
        # 2. Split training data, validation data, and test data according to the label #
        # ============================================================================= #
        anomaly_mask = targets == args.anomaly_class
        anomaly_index = torch.nonzero(anomaly_mask, as_tuple=True)
        normal_index = torch.nonzero(~anomaly_mask, as_tuple=True)
        anomaly_data = data[anomaly_index]
        anomaly_data = anomaly_data.unsqueeze(1)
        anomaly_targets = targets[anomaly_index]
        normal_data = data[normal_index]
        normal_data = normal_data.unsqueeze(1)
        normal_targets = targets[normal_index]
        # shuffle
        normal_shuffle_idx = torch.randperm(normal_data.shape[0])
        anomaly_shuffle_idx = torch.randperm(anomaly_data.shape[0])
        normal_data = normal_data[normal_shuffle_idx]
        normal_targets = normal_targets[normal_shuffle_idx]
        anomaly_data = anomaly_data[anomaly_shuffle_idx]
        anomaly_targets = anomaly_targets[anomaly_shuffle_idx]
        # split
        train_length = int(len(normal_data) * train_ratio)
        valid_length = int(len(normal_data) * valid_ratio)
        test_length = len(normal_data) - train_length - valid_length
        train_data, valid_data, test_data = torch.split(normal_data, [train_length, valid_length, test_length])
        train_targets, valid_targets, test_targets = torch.split(normal_targets, [train_length, valid_length, test_length])
        # ============================================================================= #
        #           3. Adjusting the balance of the anomaly ratio of the test           #
        #               set of normal data and the data of abnormal data                #
        # ============================================================================= #
        current_ratio = anomaly_data.shape[0] / (test_length + anomaly_data.shape[0])
        if current_ratio < args.anomaly_ratio:
            normal_count = int(len(anomaly_data) / args.anomaly_ratio - len(anomaly_data))
            normal_index = torch.randperm(test_length)[:normal_count]
            test_data = test_data[normal_index]
            test_targets = test_targets[normal_index]
        elif current_ratio > args.anomaly_ratio:
            anomaly_count = int(test_length * args.anomaly_ratio / (1 - args.anomaly_ratio))
            anomaly_index = torch.randperm(anomaly_data.shape[0])[:anomaly_count]
            anomaly_data = anomaly_data[anomaly_index]
            anomaly_targets = anomaly_targets[anomaly_index]
        else:
            pass
        # ================================================================== #
        #                         4. concatenate data                        #
        # ================================================================== #
        test_data = torch.cat((test_data, anomaly_data), 0)
        test_targets = torch.cat((test_targets, anomaly_targets), 0)
        # ================================================================== #
        #                           5. make dataset                          #
        # ================================================================== #
        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_targets)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_targets)
        return train_dataset, valid_dataset, test_dataset

    def _get_data_loader(self, args, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return dataloader
    
    def _get_transform(self, args, is_target):
        transforms_list = []
        if not is_target:
            if args.random_crop:
                transforms_list.append(transforms.RandomCrop(args.crop_size))
            if args.grayscale:
                transforms_list.append(transforms.Grayscale())
                args.channel_num = 1
            transforms_list.append(transforms.ToTensor())
            if args.normalize:
                if args.channel_num == 3:
                    transforms_list.append(transforms.Normalize((0.5,0.5,0.5,), (0.5,0.5,0.5,)))
                elif args.channel_num == 1:
                    transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            pass
        return transforms.Compose(transforms_list)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.image_paths = []
        self.targets = []
        if(train):
            root_dir = os.path.join(root, 'train')
        else:
            root_dir = os.path.join(root, 'test')
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            images_in_folder = sorted(os.listdir(folder_path))
            self.image_paths.extend([os.path.join(folder_path, image) for image in images_in_folder])
            if folder == 'good':
                self.targets.extend([0]*len(images_in_folder))
            else:
                self.targets.extend([1]*len(images_in_folder))
        assert len(self.image_paths) == len(self.targets), 'images and label number mismatch'
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        x = Image.open(self.image_paths[idx])
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

class customTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors, transform=None, target_transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), 'tensor dimension mismatch'
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[0]
        y = self.tensors[1]
        x = self.transform(x)
        y = self.target_transform(y)
        return tuple(x, y)

    def __len__(self):
        return self.tensors[0].size(0)

if __name__ == "__main__":
    from torchvision.utils import save_image
    args = parser.get_args()
    print('running config :', args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    USE_CUDA = torch.cuda.is_available()
    
    # Reproducibility
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using Device: ",DEVICE)

    data_loader = DataLoader(args)
    for batch_idx, (batch_imgs, batch_label) in enumerate(data_loader.train_data_loader):
        batch_imgs = batch_imgs.to(device=DEVICE, dtype=torch.float)
        batch_label = batch_label.to(device=DEVICE, dtype=torch.float)
        for i in range(len(batch_imgs)):
            print("shape: ", batch_imgs[i].shape, "label: ", batch_label[i])
            # save_image(batch_imgs[i].double(), "/root/anomaly_detection/temp/" + str(batch_idx) + "_" + str(i) + "_" + str(batch_label[i]) + ".png", "PNG")
    for batch_idx, (batch_imgs, batch_label) in enumerate(data_loader.valid_data_loader):
        for i in range(len(batch_imgs)):
            print("shape: ", batch_imgs[i].shape, "label: ", batch_label[i])
            # save_image(batch_imgs[i].double(), "/root/anomaly_detection/temp/" + str(batch_idx) + "_" + str(i) + "_" + str(batch_label[i]) + ".png", "PNG")
    for batch_idx, (batch_imgs, batch_label) in enumerate(data_loader.test_data_loader):
        for i in range(len(batch_imgs)):
            print("shape: ", batch_imgs[i].shape, "label: ", batch_label[i])
            # save_image(batch_imgs[i].double(), "/root/anomaly_detection/temp/" + str(batch_idx) + "_" + str(i) + "_" + str(batch_label[i]) + ".png", "PNG")