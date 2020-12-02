import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import os
import sys
import random
from pathlib import Path

DATASET_PATH = '/hd/gms/'

__all__ = ['objs', 'set_root_path',
           'get_x', 'get_x_standardized',
           'detection_auroc', 'segmentation_auroc']

objs = ['circle_hole', 'rect_hole']


def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def bilinears(images, shape) -> np.ndarray:
    import cv2
    N = images.shape[0]
    new_shape = (N,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(N):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret


def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    # print(images.shape)
    return images


def set_root_path(new_path):
    global DATASET_PATH
    DATASET_PATH = new_path


def get_x(obj, mode='train'):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/{mode}/*/*.png')
    fpaths = sorted(glob(fpattern))

    if mode == 'test':
        fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
        fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

        images1 = np.asarray(list(map(imread, fpaths1)))
        images2 = np.asarray(list(map(imread, fpaths2)))
        images = np.concatenate([images1, images2])

    else:
        # Use this code when split dataset into train, valid and test only once before training start
        # train : valid : test = 6 : 3 : 1

        # num_test = int(len(fpaths)*0.1)
        # num_valid = int(len(fpaths)*0.3)
        # random.shuffle(fpaths)

        # test_files = fpaths[:num_test]
        # print('Num of test_files :', len(test_files))
        # for f in test_files:
        #     Path(f).rename(os.path.join(f'/hd/gms/{obj}/test/good/', os.path.basename(f)))
        
        # valid_files = fpaths[num_test:num_test+num_valid] 
        # print('Num of valid_files :', len(valid_files))
        # for f in valid_files:
        #     Path(f).rename(os.path.join(f'/hd/gms/{obj}/valid/good/', os.path.basename(f)))

        # train_files = fpaths[num_test+num_valid:]
        # print('Num of train_files :', len(train_files))
        # sys.exit()
        
        images = np.asarray(list(map(imread, fpaths)))

    # if images.shape[-1] != 3:
    #     print(images.shape)
    #     print('Change gray to rgb')
    #     images = gray2rgb(images)
    images = list(map(resize, images))
    images = np.asarray(images)
    return images


def get_x_standardized(obj, mode='train'):
    x = get_x(obj, mode=mode)
    mean = get_mean(obj)
    return (x.astype(np.float32) - mean) / 255


def get_label(obj):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/test/*/*.png')
    fpaths = sorted(glob(fpattern))
    fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
    fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

    Nanomaly = len(fpaths1)
    Nnormal = len(fpaths2)
    labels = np.zeros(Nanomaly + Nnormal, dtype=np.int32)
    labels[:Nanomaly] = 1
    return labels


def get_mask(obj):
    fpattern = os.path.join(DATASET_PATH, f'{obj}/ground_truth/*/*.png')
    fpaths = sorted(glob(fpattern))
    masks = np.asarray(list(map(lambda fpath: resize(imread(fpath), (256, 256)), fpaths)))
    Nanomaly = masks.shape[0]
    Nnormal = len(glob(os.path.join(DATASET_PATH, f'{obj}/test/good/*.png')))

    masks[masks <= 128] = 0
    masks[masks > 128] = 255
    results = np.zeros((Nanomaly + Nnormal,) + masks.shape[1:], dtype=masks.dtype)
    results[:Nanomaly] = masks

    return results


def get_mean(obj):
    images = get_x(obj, mode='train')
    mean = images.astype(np.float32).mean(axis=0)
    return mean


def detection_auroc(obj, anomaly_scores):
    label = get_label(obj)  # 1: anomaly 0: normal
    auroc = roc_auc_score(label, anomaly_scores)
    return auroc


def segmentation_auroc(obj, anomaly_maps):
    gt = get_mask(obj)
    gt = gt.astype(np.int32)
    gt[gt == 255] = 1  # 1: anomaly

    anomaly_maps = bilinears(anomaly_maps, (256, 256))
    auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
    return auroc

def _classification_report(anomaly_scores, obj, threshold, target_label):
    label = get_label(obj)
    print('label: ', label)
    try:
        pred = anomaly_scores > threshold
        print('pred: ', pred)
        # pred = pred.astype(int)
        # pred[pred==1] = -1
        # pred[pred==0] = 1
        print(confusion_matrix(label, pred, [1, 0]))
        return classification_report(label, pred, [1, 0], ['Anomaly', 'Normal'])
    except Exception as e:
        print(e)
