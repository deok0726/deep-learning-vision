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
        self.train_data_loader = self._get_data_loader(args, train_dataset, args.train_batch_size)
        self.valid_data_loader = self._get_data_loader(args, valid_dataset, args.train_batch_size)
        self.test_data_loader = self._get_data_loader(args, test_dataset, args.test_batch_size)
        print("train data size: ", len(self.train_data_loader)*args.train_batch_size)
        print("valid data size: ", len(self.valid_data_loader)*args.train_batch_size)
        print("test data size: ", len(self.test_data_loader)*args.test_batch_size)

    def _get_datasets(self, args, source_transform, target_transform):
        if args.dataset_name == 'MNIST':
            train_dataset = MNIST(root=args.dataset_root, train=True, download=True, transform=source_transform, target_transform=target_transform)
            test_dataset = MNIST(root=args.dataset_root, train=False, download=True, transform=source_transform, target_transform=target_transform)
            # test_dataset = MNIST(root=args.dataset_root, train=False, download=True, transform=transforms.ToTensor(), target_transform=target_transform)
            train_dataset, valid_dataset, test_dataset = self._preprocess_to_anomaly_detection_dataset(args, train_dataset, test_dataset)
        elif args.dataset_name == 'FMNIST':
            train_dataset = FashionMNIST(root=args.dataset_root, train=True, download=True, transform=source_transform, target_transform=target_transform)
            test_dataset = FashionMNIST(root=args.dataset_root, train=False, download=True, transform=source_transform, target_transform=target_transform)
            # test_dataset = FashionMNIST(root=args.dataset_root, train=False, download=True, transform=transforms.ToTensor(), target_transform=target_transform)
            train_dataset, valid_dataset, test_dataset = self._preprocess_to_anomaly_detection_dataset(args, train_dataset, test_dataset)
        elif args.dataset_name == 'MvTec':
            train_dataset = ImageDataset(root=args.dataset_root, train=True, transform=source_transform, target_transform=target_transform)
            test_dataset = ImageDataset(root=args.dataset_root, train=False, transform=source_transform, target_transform=target_transform)
            # test_dataset = ImageDataset(root=args.dataset_root, train=False, transform=transforms.ToTensor(), target_transform=target_transform)
            test_dataset = ImageDataset(root=args.dataset_root, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5,), (0.5,0.5,0.5,))]), target_transform=target_transform)
            train_length = int(len(train_dataset) * (args.train_ratio / (args.train_ratio + args.valid_ratio)))
            valid_length = len(train_dataset) - train_length
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
        data = torch.cat((original_train_dataset.data, original_test_dataset.data))
        targets = torch.cat((original_train_dataset.targets, original_test_dataset.targets))
        # ============================================================================= #
        # 2. Split training data, validation data, and test data according to the label #
        # ============================================================================= #
        if args.unique_anomaly:
            anomaly_mask = targets == args.target_label
        else:
            anomaly_mask = targets != args.target_label
        anomaly_index = torch.nonzero(anomaly_mask, as_tuple=True)
        normal_index = torch.nonzero(~anomaly_mask, as_tuple=True)
        anomaly_data = data[anomaly_index]
        anomaly_targets = targets[anomaly_index]
        normal_data = data[normal_index]
        normal_targets = targets[normal_index]
        # shuffle
        if args.shuffle:
            normal_shuffle_idx = np.random.permutation(normal_data.shape[0])
            anomaly_shuffle_idx = np.random.permutation(anomaly_data.shape[0])
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
            test_data = test_data[:normal_count]
            test_targets = test_targets[:normal_count]
        elif current_ratio > args.anomaly_ratio:
            anomaly_count = int(test_length * args.anomaly_ratio / (1 - args.anomaly_ratio))
            anomaly_data = anomaly_data[:anomaly_count]
            anomaly_targets = anomaly_targets[:anomaly_count]
        elif args.anomaly_ratio == -1:
            pass
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
        train_dataset = customTensorDataset(train_data, train_targets, original_train_dataset.transform, original_train_dataset.target_transform)
        valid_dataset = customTensorDataset(valid_data, valid_targets, original_test_dataset.transform, original_test_dataset.target_transform)
        test_dataset = customTensorDataset(test_data, test_targets, original_test_dataset.transform, original_test_dataset.target_transform)
        return train_dataset, valid_dataset, test_dataset

    def _get_data_loader(self, args, dataset, batch_size):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=True)
        return dataloader
    
    def _get_transform(self, args, is_target):
        transforms_list = []
        if not is_target:
            if args.random_crop:
                transforms_list.append(transforms.RandomCrop(args.crop_size))
            if args.resize:
                transforms_list.append(transforms.Resize(args.resize_size))
            if args.grayscale:
                transforms_list.append(transforms.Grayscale())
                args.channel_num = 1
            if args.random_rotation:
                transforms_list.append(transforms.RandomRotation(360))
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
    def __init__(self, data, target, transform=None, target_transform=None):
        assert data.shape[0] == target.shape[0], 'tensor dimension mismatch'
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.data[index]
        target = int(self.target[index])
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.data.shape[0]

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