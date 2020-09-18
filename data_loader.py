import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import argument_parser as parser
import random
import numpy as np


class DataLoader:

    def __init__(self, args):
        source_transform_list = self._get_transform(args, is_target=False)
        target_transform_list = self._get_transform(args, is_target=True)
        train_dataset, valid_dataset, test_dataset = self._get_datasets(args, source_transform=source_transform_list, target_transform=target_transform_list)
        self.train_data_loader = self._get_data_loader(args, train_dataset)
        self.valid_data_loader = self._get_data_loader(args, valid_dataset)
        self.test_data_loader = self._get_data_loader(args, test_dataset)

    def _get_datasets(self, args, source_transform, target_transform):
        if args.dataset_name == 'MNIST':
            train_dataset = MNIST(root=args.dataset_root, train=True, download=True, transform=source_transform, target_transform=target_transform)
            test_dataset = MNIST(root=args.dataset_root, train=False, download=True, transform=source_transform, target_transform=target_transform)
            train_dataset, valid_dataset, test_dataset = self._preprocess_to_anomaly_detection_dataset(args, train_dataset, test_dataset)
        return train_dataset, valid_dataset, test_dataset
    
    def _preprocess_to_anomaly_detection_dataset(self, args, original_train_dataset, original_test_dataset):
        # Divide normal data according to the ratio of training set: validation set: test set
        # 1. Combine original training set and test set
        total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
        train_ratio = args.train_ratio / total_ratio
        valid_ratio = args.valid_ratio / total_ratio
        test_ratio = args.test_ratio / total_ratio
        train_data, train_targets = original_train_dataset.data, original_train_dataset.targets
        test_data, test_targets = original_test_dataset.data, original_test_dataset.targets
        data = torch.cat((train_data, test_data))
        targets = torch.cat((train_targets, test_targets))
        # 2. Split training set, validation set, and test set according to the label
        anomaly_mask = targets == args.anomaly_class
        anomaly_index = torch.nonzero(anomaly_mask)
        normal_index = torch.nonzero(~anomaly_mask)
        anomaly_data = data[anomaly_index]
        anomaly_targets = targets[anomaly_index]
        normal_data = data[normal_index]
        normal_targets = targets[normal_index]
        anomaly_dataset = torch.utils.data.TensorDataset(anomaly_data, anomaly_targets)
        normal_dataset = torch.utils.data.TensorDataset(normal_data, normal_targets)
        train_length = int(len(normal_dataset) * train_ratio)
        valid_length = int(len(normal_dataset) * valid_ratio)
        test_length = len(normal_dataset) - train_length - valid_length
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(normal_dataset, [train_length, valid_length, test_length])
        # 3. Adjusting the balance of the anomaly ratio of the test set of normal data and the data set of abnormal data set
        current_ratio = len(anomaly_dataset) / (test_length + len(anomaly_dataset))
        if current_ratio < args.anomaly_ratio:
            normal_count = int(len(anomaly_dataset) / args.anomaly_ratio - len(anomaly_dataset))
            test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(test_length)[:normal_count])
            print(len(test_dataset))
        elif current_ratio > args.anomaly_ratio:
            anomaly_count = int(test_length * args.anomaly_ratio / (1 - args.anomaly_ratio))
            anomaly_dataset = torch.utils.data.Subset(anomaly_dataset, torch.randperm(len(anomaly_dataset))[:anomaly_count])
            print(len(anomaly_dataset))
        else:
            pass
        # 4. concatenate dataset
        test_dataset = torch.utils.data.ConcatDataset((test_dataset, anomaly_dataset))
        # 5. apply transforms
        train_dataset.transform = original_train_dataset.transform
        train_dataset.target_transform = original_train_dataset.target_transform
        valid_dataset.transform = original_train_dataset.transform
        valid_dataset.target_transform = original_train_dataset.target_transform
        test_dataset.transform = original_test_dataset.transform
        test_dataset.target_transform = original_test_dataset.target_transform
        return train_dataset, valid_dataset, test_dataset

    def _get_data_loader(self, args, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return dataloader
    
    def _get_transform(self, args, is_target):
        transforms_list = []
        if not is_target:
            if args.grayscale:
                transforms_list.append(transforms.Grayscale())
            transforms_list.append(transforms.ToTensor())
        else:
            pass
        return transforms.Compose(transforms_list)


if __name__ == "__main__":
    from torchvision.utils import save_image
    args = parser.get_args()
    # Reproducibility
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    data_loader = DataLoader(args)
    dataiter = iter(data_loader.train_data_loader)
    for batch_ndx, sample in enumerate(data_loader.train_data_loader):
        for i in range(len(sample[0])):
            print("shape: ", sample[0][i].shape, "label: ", sample[1][i])
            save_image(sample[0][i].double(), "/root/anomaly_detection/temp/" + str(batch_ndx) + "_" + str(i) + "_" + str(sample[1][i]) + ".png", "PNG")
    for batch_ndx, sample in enumerate(data_loader.valid_data_loader):
        for i in range(len(sample[0])):
            print("shape: ", sample[0][i].shape, "label: ", sample[1][i])
            save_image(sample[0][i].double(), "/root/anomaly_detection/temp/" + str(batch_ndx) + "_" + str(i) + "_" + str(sample[1][i]) + ".png", "PNG")
    for batch_ndx, sample in enumerate(data_loader.test_data_loader):
        for i in range(len(sample[0])):
            print("shape: ", sample[0][i].shape, "label: ", sample[1][i])
            save_image(sample[0][i].double(), "/root/anomaly_detection/temp/" + str(batch_ndx) + "_" + str(i) + "_" + str(sample[1][i]) + ".png", "PNG")