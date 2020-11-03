import argparse
import os

parser = argparse.ArgumentParser(description='anomaly_detection')

# main
parser.add_argument('--random_seed', type=int, default=0, help="random seed")
parser.add_argument('--model_name', type=str, default='AE', help="model name")
parser.add_argument('--num_epoch', type=int, default=10, help="num_epoch")
parser.add_argument('--train_batch_size', type=int, default=16, help="train batch size")
parser.add_argument('--test_batch_size', type=int, default=4, help="test batch size")
parser.add_argument('--learning_rate', type=float, default='0.001', help="learning rate")
parser.add_argument('--learning_rate_decay', action='store_true', default=False, help="apply learning rate decay")
parser.add_argument('--learning_rate_decay_ratio', type=float, default='0.5', help="learning rate decay ratio")
parser.add_argument('--end_learning_rate', type=float, default='0.0001', help="learning rate")
parser.add_argument('--train', action='store_true', default=False, help="start training")
parser.add_argument('--test', action='store_true', default=False, help="start testing")
parser.add_argument('--exp_name', type=str, default='temp', help="experiment name")
parser.add_argument('--train_tensorboard_shown_image_num', type=int, default=4, help="The number of the train and valid images shown in tensorboard")
parser.add_argument('--test_tensorboard_shown_image_num', type=int, default=1, help="The number of the test images shown in tensorboard")
parser.add_argument('--target_label', type=int, default=0, help="target label")
parser.add_argument('--unique_anomaly', action='store_true', default=False, help="Unique anomaly class")
parser.add_argument('--reproducibility', action='store_true', default=False, help="Reproducibility On")
parser.add_argument('--save_result_images', action='store_true', default=False, help="saving result images")

# data loader
parser.add_argument('--dataset_name', type=str, default='MNIST', help="dataset name")
parser.add_argument('--dataset_root', type=str, default='/hd/', help="dataset name")
parser.add_argument('--channel_num', type=int, default=3, help="dataset name")
parser.add_argument('--num_workers', type=int, default=0, help="number of data loader workers")
parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle or not")
parser.add_argument('--grayscale', action='store_true', default=False, help="Grayscale transform")
parser.add_argument('--normalize', action='store_true', default=False, help="Normalize transformation")
parser.add_argument('--random_rotation', action='store_true', default=False, help="Random Rotation transformation")
parser.add_argument('--random_crop', action='store_true', default=False, help="Random Crop transformation")
parser.add_argument('--crop_size', type=int, default=300, help="Cropped image size")
parser.add_argument('--resize', action='store_true', default=False, help="Resize transformation")
parser.add_argument('--resize_size', type=int, default=300, help="Cropped image size")
parser.add_argument('--train_ratio', type=float, default=0.6, help="training dataset ratio")
parser.add_argument('--valid_ratio', type=float, default=0.3, help="validation dataset ratio")
parser.add_argument('--test_ratio', type=float, default=0.1, help="test dataset ratio")
parser.add_argument('--anomaly_ratio', type=float, default=0.3, help="anomaly data ratio")

# save
parser.add_argument('--checkpoint_dir', type=str, default='/hd/checkpoints/', help="ckpt dir")
parser.add_argument('--tensorboard_dir', type=str, default='/hd/tensorboard_logs/', help="ckpt dir")

# model args
parser.add_argument('--entropy_loss_coef', type=float, default=0.0002, help="entropy loss weight for memae model")

def get_args():
    args = parser.parse_args()
    directories = [args.checkpoint_dir,
    args.tensorboard_dir,
    os.path.join(args.checkpoint_dir, args.model_name),
    os.path.join(args.tensorboard_dir, args.model_name),
    os.path.join(os.path.join(args.checkpoint_dir, args.model_name), args.exp_name),
    os.path.join(os.path.join(args.tensorboard_dir, args.model_name), args.exp_name),
    os.path.join(os.path.join(os.path.join(args.tensorboard_dir, args.model_name), args.exp_name), 'test_results')
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return args