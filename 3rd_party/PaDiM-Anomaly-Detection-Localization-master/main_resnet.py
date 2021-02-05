import random
from random import sample
import argparse
import numpy as np
import os
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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
# from efficientnet_pytorch import EfficientNet
import datasets.daejoo as daejoo
import sys
import time

# device setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)
# sys.exit()

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--data_path", type=str, default="/root/daejoo/")
    parser.add_argument("--save_path", type=str, default="/root/Tmax/3rd_party/PaDiM-Anomaly-Detection-Localization-master/daejoo_result")
    parser.add_argument("--arch", type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    parser.add_argument("--dim", type=int, default='448')
    return parser.parse_args()

def main_nomask():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        # t_d = 3   #layer1
        # d = 3
        # t_d =    #layer2
        # d = 
        t_d = 448   #layer3
        d = 448
        # t_d = 960   #layer4
        # d = 960
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        # t_d =    #layer1
        # d = 
        # t_d = 768   #layer2
        # d = 768
        t_d = 1792  #layer3
        d = 1792
            
    print('dimension is', d)
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)
    
    idx = torch.tensor(sample(range(0, t_d), d))
    
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        # print('hook is called')
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # model.layer1[-1].register_forward_hook(hook)
    # model.layer2[-1].register_forward_hook(hook)

    # model.layer1[-1].register_forward_hook(hook)
    # model.layer2[-1].register_forward_hook(hook)
    # model.layer3[-1].register_forward_hook(hook)
    # model.layer4[-1].register_forward_hook(hook)

    new_path = os.path.join(args.save_path, 'temp_final_%s' % args.arch)
    os.makedirs(new_path, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in daejoo.CLASS_NAMES:

        train_dataset = daejoo.MVTecDataset_nomask(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=128, pin_memory=True)
        # train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = daejoo.MVTecDataset_nomask(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=128, pin_memory=True)
        # test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # train_outputs = OrderedDict([('layer1', [])])
        # test_outputs = OrderedDict([('layer1', [])])

        # train_outputs = OrderedDict([('layer1', []), ('layer2', [])])
        # test_outputs = OrderedDict([('layer1', []), ('layer2', [])])

        # train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
        # test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_final_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            # hand images and labels
            for (x, y) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction & hook function call
                # print(x.size())
                # print(torch.mean(x, (0, 2, 3)))
                # print(torch.std(x, (0, 2, 3)))
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                    # print('train_outputs[%s]: ' %k, outputs)
                # initialize hook outputs
                outputs = []
            # print('finished train_outputs.append')
            for k, v in train_outputs.items():
                # print(len(train_outputs[k]))
                train_outputs[k] = torch.cat(v, 0)
            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
            # for layer_name in ['layer2']:
            # for layer_name in ['layer2', 'layer3', 'layer4']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
                print('finished embedding concat:', layer_name)

            # randomly select d dimension
            print(embedding_vectors.size())
            # sys.exit()
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                # pickle.dump(train_outputs, f)
                pickle.dump(train_outputs, f, protocol=4)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        # gt_mask_list = []
        test_imgs = []

        # extract test set features
        ####################################################################################start inference######################################################################################
        print('start inference')
        start = time.time()
        
        for (x, y) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            # gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        # for k in test_outputs.keys():
        #     print(len(test_outputs[k]))

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        
        for layer_name in ['layer2', 'layer3']:
        # for layer_name in ['layer2']:
        # for layer_name in ['layer2', 'layer3', 'layer4']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
            print('finished embedding concat:', layer_name)

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        print(B, C, H, W)
        embedding_vectors = embedding_vectors.view(B, C, H * W).detach().numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            
            # for in_sample in embedding_vectors:
            #     print('in_sample: ', in_sample.shape)
            #     print('mean: ', mean.shape)
            #     print('conv_inv: ', conv_inv.shape)
            #     dist = [mahalanobis(in_sample[:, i], mean, conv_inv)]
            
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        # print('img_scores is :', img_scores)
        gt_list = np.asarray(gt_list)
        # print('gt_list is :', gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        
    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    end = time.time()
    ####################################################################################finish inference######################################################################################
    print('inference time:', end-start)
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    # print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    # fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'temp_%s' % args.arch, 'roc_curve.png'), dpi=100)


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    # print('x.size():', x.size())
    # print('x size:', x.element_size() * x.nelement())
    # print('B, C1, H1, W1: ', B, C1, H1, W1)
    
    B2, C2, H2, W2 = y.size()
    # print('y.size():', y.size())
    # print('y size:', y.element_size() * y.nelement())
    # print('B2, C2, H2, W2: ', B2, C2, H2, W2)

    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    # print('x.unfold size:', x.size())
    x = x.view(B, C1, -1, H2, W2)
    # print('x.view size:', x.size())
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    # print('z.zeros size:', z.size())
    # print('z size:', z.element_size() * z.nelement())

    # print('x.size():', x.size())
    # print('x.size(0): ', x.size(0))
    # print('x.size(1): ', x.size(1))
    # print('x.size(2): ', x.size(2))
    # print('x.size(3): ', x.size(3))
    # print('x.size(4): ', x.size(4))

    for i in range(x.size(2)):
        # print(i)
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)

        # print('size:', sys.getsizeof(z.storage()))
        # print('size:', z.element_size() * z.nelement())
        # print('z size:', z.size())
    print('z.size():', z.size())
    z = z.view(B, -1, H2 * W2)
    print('z.view size:', z.size())
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    # print('z.fold size:', z.size())
    return z


if __name__ == '__main__':
    # main()
    main_nomask()
