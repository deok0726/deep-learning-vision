import argparse
import os

parser = argparse.ArgumentParser(description='anomaly_detection')

# main
parser.add_argument('--model_name', type=str, default='AE', help="model name")
parser.add_argument('--num_epoch', type=int, default=10, help="num_epoch")
parser.add_argument('--batch_size', type=int, default=50, help="batch size")
parser.add_argument('--learning_rate', type=float, default='0.001', help="learning rate")
parser.add_argument('--train', action='store_true', default=False, help="start training")
parser.add_argument('--exp_name', type=str, default='temp', help="experiment name")
parser.add_argument('--tensorboard_shown_image_num', type=int, default=4, help="The number of the images shown in tensorboard")

# data loader
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
parser.add_argument('--dataset_name', type=str, default='MNIST', help="dataset name")
parser.add_argument('--dataset_root', type=str, default='/hd/', help="dataset name")
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

# save
parser.add_argument('--checkpoint_dir', type=str, default='/hd/checkpoints/', help="ckpt dir")
parser.add_argument('--tensorboard_dir', type=str, default='/hd/tensorboard_logs/', help="ckpt dir")

def get_args():
    args = parser.parse_args()
    directories = [args.checkpoint_dir,
    args.tensorboard_dir,
    os.path.join(args.checkpoint_dir, args.model_name),
    os.path.join(args.tensorboard_dir, args.model_name),
    os.path.join(os.path.join(args.checkpoint_dir, args.model_name), args.exp_name),
    os.path.join(os.path.join(args.tensorboard_dir, args.model_name), args.exp_name)
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return args