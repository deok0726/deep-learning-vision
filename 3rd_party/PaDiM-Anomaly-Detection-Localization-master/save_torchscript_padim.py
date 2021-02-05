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

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument("--data_path", type=str, default="/root/daejoo_final/")
parser.add_argument('--save_path', default='/root/Tmax/3rd_party/PaDiM-Anomaly-Detection-Localization-master/daejoo_result/', type=str)
parser.add_argument('--class_name', default='orange', type=str)
parser.add_argument('--exp_name', default='torchscript', type=str)

class Inferece_PaDiM(torch.nn.Module):
    def __init__(self,  model, index):
        super(Inferece_PaDiM, self).__init__()
        self.model = model
        self.idx = index
        self.USE_CUDA = torch.cuda.is_available()
        self.outputs = []

        self.model.layer1[-1].register_forward_hook(self.hook)
        self.model.layer2[-1].register_forward_hook(self.hook)
        self.model.layer3[-1].register_forward_hook(self.hook)

        # Select Device
        self.DEVICE = torch.device("cuda" if self.USE_CUDA else "cpu")
        print("Using Device: ", self.DEVICE)

    def embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()
        B2, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)

        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)

        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
        return z

    def hook(self, module, input, output):
        print('######################hook is called############################')
        self.outputs.append(output)
    
    def forward(self, data):
        print('****************************start forward********************************')
        # Load dataset
        # test_dataset = daejoo_one.MVTecDataset_nomask(data, class_name=args.class_name, is_train=False)
        # test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        gt_list = []
        test_imgs = []

        # Load model & optimizer
        self.model.to(self.DEVICE)
        self.model.eval()
        random.seed(1024)
        torch.manual_seed(1024)
        if self.USE_CUDA:
            torch.cuda.manual_seed_all(1024)

        # Track the checkpoint directory
        checkpoint = os.path.join(args.save_path, 'temp448_%s' % args.model,  'train_%s.pkl' % args.class_name)
        if os.path.exists(checkpoint):
        #    print ('train_%s.pkl' % args.class_name, 'EXISTS')
            print('load train set feature from: %s' % checkpoint)
            with open(checkpoint, 'rb') as f:
                train_outputs = pickle.load(f)
        else:
            raise FileNotFoundError

        # outputs = []

        print('start inference')
        start = time.time()
        
        # for (x, _) in tqdm(test_dataloader, '| feature extraction | inference | %s |' % args.class_name):
        #     # test_imgs.extend(x.cpu().detach().numpy())
        #     # gt_list.extend(y.cpu().detach().numpy())

        #     # model prediction
        #     print(type(x))
        #     sys.exit()
        #     with torch.no_grad():
        #         _ = model(x.to(self.DEVICE))
        
        #     # get intermediate layer outputs
        #     for k, v in zip(test_outputs.keys(), outputs):
        #         test_outputs[k].append(v.cpu().detach())
        
        #     # initialize hook outputs
        #     # outputs = []
        x = data
        with torch.no_grad():
            _ = model(x.to(self.DEVICE))
        
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), self.outputs):
            test_outputs[k].append(v.cpu().detach())

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        self.outputs = []

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])
            print('finished embedding concat:', layer_name)

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        print("B, C, H, W are ", B, C, H, W)
        embedding_vectors = embedding_vectors.view(B, C, H * W).detach().numpy()
        dist_list = []
        # for i in tqdm(range(H * W)):
        for i in tqdm(range(2 * 2)):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]       

            # for sample in embedding_vectors:
            #     print(sample[:,i].size)
            #     print(mean.size)
            #     dist = mahalanobis(sample[:, i], mean, conv_inv)

            dist_list.append(dist)

        # print(dist_list.size)
        # print(dist_list.shape)
        # dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, 2, 2)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # score_map_tensor = torch.from_numpy(score_map)        

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        # print('scores type is ', type(scores))
        # scores_tensor = torch.from_numpy(scores)
        # print('scores_tensor type is ', scores_tensor)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        # print("img_scores type is ", type(img_scores))

        # gt_list = np.asarray(gt_list)
        # fpr, tpr, _ = roc_curve(gt_list, img_scores)
        # img_roc_auc = roc_auc_score(gt_list, img_scores)
        # total_roc_auc.append(img_roc_auc)
        # print('image ROCAUC: %.3f' % (img_roc_auc))
        # fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
            
            
        # print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
        end = time.time()
        print('inference time:', end-start)
        # print("@@@@@@@@@@@@@@@@@@@@dist_list@@@@@@@@@@@@@@@@@@@@@@", dist_list)
        return dist_list


def automkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+' generated')
    else:
        print(path+' exists')

def load_model(args):
    if args.model=='resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 448
        index = torch.tensor(sample(range(0, t_d), d))
    elif args.model == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 1792
        index = torch.tensor(sample(range(0, t_d), d))
    else:
        raise NotImplementedError
    return model, index


if __name__ == "__main__":

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    args = parser.parse_args()

    # Load model
    model, index = load_model(args)

    # Make one random input
    data = torch.rand(1, 3, 224, 224, requires_grad=True)
    
    # Make padim class
    # padim = Inferece_PaDiM(data, model, index)
    padim = Inferece_PaDiM(model, index)

    # Test padim with random input
    # print('############################before jit.trace#######################################')
    traced_cell = torch.jit.trace(padim, data)
    # padim(data)
    print('#########################Finished trace############################')
    print(traced_cell)
    sys.exit()
    # torchscript_model_save_path = os.path.join("/hd/torchscript_model/", args.model_name)
    torchscript_model_save_path = os.path.join(args.save_path, 'temp448_%s' % args.model)
    print('torchscript model save_path is ', torchscript_model_save_path)
    # automkdir(torchscript_model_save_path)

    # torchscript_model_save_path = os.path.join(torchscript_model_save_path, args.exp_name)
    # automkdir(torchscript_model_save_path)
    # os.chdir('/root/Anomaly-Detection-PatchSVDD-PyTorch/PatchSVDD/ckpts/torchscript/')
    traced_cell.save(os.path.join(torchscript_model_save_path, 'padim_%s_torchscript_%s.pt' % (args.model, args.class_name)))
    loaded = torch.jit.load(os.path.join(torchscript_model_save_path, 'padim_%s_torchscript_%s.pt' % (args.model, args.class_name)))
    
    print('Conversion finished')
