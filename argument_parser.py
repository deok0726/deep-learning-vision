import argparse
import os

parser = argparse.ArgumentParser(description='anomaly_detection')

# main
parser.add_argument('--backbone_name', type=str, default='AE', help="model name")
parser.add_argument('--num_epoch', type=int, default='10', help="num_epoch")
parser.add_argument('--batch_size', type=int, default='50', help="batch size")
parser.add_argument('--learning_rate', type=float, default='0.001', help="learning rate")

# data loader

# train

# save
parser.add_argument('--save_dir', type=str, default='ckpt/', help="ckpt dir")

def get_args():
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args