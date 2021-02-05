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

from random import sample
import numpy as np
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import time
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18
from datasets import daejoo_one
from collections import OrderedDict

import timm

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='vit_small_patch16_224', type=str)
# parser.add_argument("--data_path", type=str, default="/root/daejoo_final/")
parser.add_argument('--save_path', default='/root/Tmax/3rd_party/PaDiM-Anomaly-Detection-Localization-master/daejoo_result/', type=str)
# parser.add_argument('--class_name', default='orange', type=str)
parser.add_argument('--exp_name', default='torchscript', type=str)

class ViT(torch.nn.Module):
    def __init__(self, model):
        super(ViT, self).__init__()
        self.model = model
        self.outputs = []

        if args.model == 'vit_small_patch16_224': # 0~7
            # print('vit_small called')
            self.model.blocks[5].register_forward_hook(self.hook)
            # print('block register called')
        if args.model == 'vit_large_patch16_224': # 0~23 # 5 11 17 23
            self.model.blocks[23].register_forward_hook(self.hook)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # print('forward is called')
        self.outputs = []
        _ = self.model(x.to(self.device))
        # print('forward is finished')
        return self.outputs

    def hook(self, module, input, output):
        print('hook is called')
        self.outputs.append(output)
        print('hook is finished')


def load_model(args):
    if args.model=='vit_small_patch16_224':
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1000)
    elif args.model == 'vit_large_patch16_224':
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=1000)
    else:
        raise NotImplementedError
    return model


if __name__ == "__main__":

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    args = parser.parse_args()

    # Load model
    model = load_model(args)
    
    if torch.cuda.is_available():
        model.cuda()

    vit = ViT(model)
    
    # Make one random input
    torch.manual_seed(1024)
    data = torch.rand(1, 3, 224, 224, requires_grad=True)
    # print(data)
    # print()
    # sys.exit()
    # Test padim with random input
    traced_cell = torch.jit.trace(vit, data, check_trace=False)
    
    print(traced_cell(data))
    # print(traced_cell.code)
    # torchscript_model_save_path = os.path.join(args.save_path, 'temp_%s' % args.model)
    # print(torchscript_model_save_path)
    
    # traced_cell.save(os.path.join(torchscript_model_save_path, '%s_torchscript.pt' % args.model))
    # loaded = torch.jit.load(os.path.join(torchscript_model_save_path, '%s_torchscript.pt' % args.model))
    # print(loaded)
    # print(loaded.code)    

    print('Conversion finished')
