import argparse
import sys
import scipy
import os
import glob
from PIL import Image
import torch
import numpy as np
from skimage import io, transform
from model import ModelFactory
from torch.autograd import Variable
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
description='Video Super Resolution pytorch implementation'

def forward_x8(lr, forward_function=None):
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            #print(v2np.shape)
            if op == 'v':
                tfnp = v2np[:, :, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 2, 4, 3)).copy()
	
            ret = Variable(torch.Tensor(tfnp).cuda())
            #ret = ret.half()

            return ret

        def _transform_back(v, op):
       		
            if op == 'v':
                tfnp = v[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v.transpose((0, 1, 3, 2)).copy()
	
            return tfnp

        
        x = [lr]
        for tf in 'v', 'h': x.extend([_transform(_x, tf) for _x in x])
       
        list_r = []
        for k in range(len(x)):
            z = x[k]
            r, _ = forward_function(z)
            r = r.data.cpu().numpy()
            if k % 4 > 1:
                    r =  _transform_back(r, 'h')
            if (k % 4) % 2 == 1:
                    r =  _transform_back(r, 'v')
            list_r.append(r)
        y = np.sum(list_r,  axis=0)/4.0
       
        y = Variable(torch.Tensor(y).cuda())
        if len(y) == 1: y = y[0]
        return y

def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='TDAN',
                    help='network architecture.')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('-t', '--test-set', metavar='NAME', type=str, default='/hd/video_super_resolution/youtube_face_test_small/',
                    help='dataset for testing.')
parser.add_argument('-mp', '--model-path', metavar='MP', type=str, default='/hd/checkpoints/tdan/TDAN/4x/',
                    help='model path.')
parser.add_argument('-sp', '--save-path', metavar='SP', type=str, default='/hd/saved_models/tdan',
                    help='saving directory path.')
args = parser.parse_args()

model_factory = ModelFactory()
model = model_factory.create_model(args.model)

dir = args.test_set
lis = sorted(glob.glob(dir + '/*'))
lis2 = sorted(os.listdir(dir))

# dir_LR = args.test_set
# lis = sorted(os.listdir(dir_LR))

model_path = os.path.join(args.model_path, 'model.pt')

if not os.path.exists(model_path):
    raise Exception('Cannot find %s.' %model_path)

model = torch.load(model_path)
model.eval()
path = args.save_path

if not os.path.exists(path):
    os.makedirs(path)

for i in range(len(lis)):
    print(lis[i])
    LR = os.path.join(lis[i], 'blur4')
    ims = sorted(os.listdir(LR))
    num = len(ims)
    image = io.imread(os.path.join(LR, ims[0]))
    row, col, ch = image.shape
    frames_lr = np.zeros((5, int(row), int(col), ch))
    for j in range(num):
        for k in range(j-2, j + 3):
            idx = k-j+2
            if k < 0:
                k = -k
            if k >= num:
                k = num - 3
            frames_lr[idx, :, :, :] = io.imread(os.path.join(LR, ims[k]))
        # start = time.time()
        frames_lr = frames_lr/255.0 - 0.5
        lr = torch.from_numpy(frames_lr).float().permute(0, 3, 1, 2)
        lr = Variable(lr.cuda()).unsqueeze(0).contiguous()
        output, _ = model(lr)
        #output = forward_x8(lr, model)
        output = (output.data + 0.5)*255
        output = quantize(output, 255)
        output = output.squeeze(dim=0)
        # elapsed_time = time.time() - start
        # print(elapsed_time)
        img_name = os.path.join(os.path.join(path, lis2[i]), ims[j])
        print(img_name)
        if not os.path.exists(os.path.join(path, lis2[i])):
            os.makedirs(os.path.join(path, lis2[i]))
        Image.fromarray(np.around(output.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)).save(img_name)
        
