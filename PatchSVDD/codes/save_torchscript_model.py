import io
import sys
import numpy as np
import random
import os
import torch
import argparse
from PIL import Image
from imageio import imread
from torch.utils.data import DataLoader
from functools import reduce

from networks import *
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='Patch_SVDD', type=str)
parser.add_argument('--checkpoint_dir', default='/home/deokhwa/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts/amber/', type=str)
parser.add_argument('--obj', default='amber', type=str)
parser.add_argument('--K', default=64, type=int)
parser.add_argument('--D', default=64, type=int)
parser.add_argument('--S', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--exp_name', default='torchscript', type=str)


def load_model(args):
    if args.model_name=='Patch_SVDD':
        model = EncoderHier(K=64, D=64).cuda()
    else:
        raise NotImplementedError
    return model


def load_optimizer(args):
    optimizer = None
    if args.model_name=='Patch_SVDD':
        enc = EncoderHier(64, args.D).cuda()
        cls_64 = PositionClassifier(64, args.D).cuda()
        cls_32 = PositionClassifier(32, args.D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        optimizer = torch.optim.Adam(params=params, lr=args.lr)

    return optimizer


def automkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+' generated')
    else:
        print(path+' exists')


def resize(image, shape=(256, 256)):
    # print(shape[::-1])
    # # print(np.transpose(image,(1,2,0)).shape)
    # # img = np.transpose(image,(1,2,0))
    image = Image.fromarray((image*255).astype(np.uint8))

    # output_np.astype(np.uint8)
    # temp = Image.fromarray(image[::-1])
    return np.array(image.resize(shape[::-1]))
    # return np.array(Image.fromarray(image).resize(shape[::-1]))


def get_x(x):
    if torch.is_tensor(x):

        image = x.detach().numpy()
        image = list(map(resize, image))
 
        image = np.asarray(image)
    else:
        image = x

    # image = list(map(resize, image))
    # image = np.asarray(image)
    return image


def get_x_standardized(x):
    x = get_x(x)
    mean = get_mean(x)
    return (x.astype(np.float32) - mean) / 255


def get_mean(x):
    mean = x.astype(np.float32).mean(axis=0)
    
    return mean


if __name__ == "__main__":
    args = parser.parse_args()
    
    CHECKPOINT_SAVE_DIR = args.checkpoint_dir
    
    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    USE_CUDA = torch.cuda.is_available()
    
    # Select Device
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using Device: ",DEVICE)
    
    # Load model & optimizer
    model = load_model(args)
    optimizer = load_optimizer(args)

    # Track the checkpoint directory
    print(os.getcwd())
    os.chdir('/root/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts')
    print(os.getcwd())
    if os.path.isfile('/root/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts/enchier.pkl'):
       print ("File exist")
    else:
        raise FileNotFoundError
    
    # Restore checkpoint
    checkpoint = torch.load('/root/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts/enchier.pkl', map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Test one random input
    x = torch.rand(1, 700, 700, 3, requires_grad=True)
    x = get_x_standardized(x)
    x = NHWC2NCHW(x)
    dataset = PatchDataset_NCHW(x, K=args.K, S=args.S)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, model.D), dtype=np.float32)  # [-1, I, J, D]
    model = model.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            torch_out = model(xs)
            # print(torch_out)
    
    traced_cell = torch.jit.trace(model, xs)
    print(traced_cell)

    # torchscript_model_save_path = os.path.join("/hd/torchscript_model/", args.model_name)
    torchscript_model_save_path = os.path.join("/root/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts/", args.exp_name)
    # automkdir(torchscript_model_save_path)

    # torchscript_model_save_path = os.path.join(torchscript_model_save_path, args.exp_name)
    # automkdir(torchscript_model_save_path)
    # os.chdir('/root/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts/torchscript/')
    traced_cell.save(os.path.join(torchscript_model_save_path, 'torchscript_model.pt'))
    loaded = torch.jit.load(os.path.join(torchscript_model_save_path, 'torchscript_model.pt'))
    
    print('Finish')