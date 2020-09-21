import argparse

parser = argparse.ArgumentParser(description='anomaly_detection')

# main
parser.add_argument('--backbone_name', type=str, default='AE', help="model name")

# data loader
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
parser.add_argument('--dataset_name', type=str, default='MNIST', help="dataset name")
parser.add_argument('--dataset_root', type=str, default='/hd', help="dataset name")
parser.add_argument('--channel_num', type=int, default=3, help="dataset name")
parser.add_argument('--num_workers', type=int, default=0, help="number of data loader workers")
parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle or not")
parser.add_argument('--grayscale', action='store_true', default=False, help="Grayscale transform")
parser.add_argument('--normalize', action='store_true', default=False, help="Normalize transformation")
parser.add_argument('--random_crop', action='store_true', default=False, help="Random Crop transformation")
parser.add_argument('--crop_size', type=int, default=300, help="Cropped image size")
parser.add_argument('--anomaly_class', type=int, default=0, help="anormaly class label")
parser.add_argument('--train_ratio', type=float, default=2*9/10, help="training dataset ratio")
parser.add_argument('--valid_ratio', type=float, default=2*1/10, help="validation dataset ratio")
parser.add_argument('--test_ratio', type=float, default=1, help="test dataset ratio")
parser.add_argument('--anomaly_ratio', type=float, default=0.1, help="anomaly data ratio")
# train
parser.add_argument('--batch_size', type=int, default=4, help="batch size")

# save

def get_args():
    args = parser.parse_args()
    return args