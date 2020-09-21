#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn

from utils.normalize import Standardizer, Rotater, Truncater


def get_norm(x, norm_type=2):
    return abs(x)**norm_type

def get_auc_roc(score, test_label):
    try:
        fprs, tprs, _ = metrics.roc_curve(test_label, score)

        return metrics.auc(fprs, tprs)
    except:
        return .0    

def get_auc_prc(score, test_label):
    try:
        precisions, recalls, _ = metrics.precision_recall_curve(test_label, score)

        return metrics.auc(recalls, precisions)
    except:
        return .0

def get_f1_score(valid_score, test_score, test_label, f1_quantiles=[.99]):
    f1s = []
    for q in f1_quantiles:
        threshold = np.quantile(valid_score, q)
        predictions = test_score > threshold
        p = (predictions & test_label).sum() / float(predictions.sum())
        r = (predictions & test_label).sum() / float(test_label.sum())

        #print((predictions & test_label).sum(), predictions.sum(), test_label.sum())
        f1s += [p * r * 2 / (p + r)]

    return f1s

def get_recon_loss(valid_diff, test_diff, test_label, f1_quantiles=[.99]):
    # if CAE
    valid_diff = valid_diff.reshape((valid_diff.shape[0], -1))
    test_diff = test_diff.reshape((test_diff.shape[0], -1))
    # test_diff = torch.flatten(test_diff, start_dim=1)
    # test_diff = test_diff.flatten(start_dim=1)

    loss = (test_diff**2).mean(axis=1)
    loss_auc_roc = get_auc_roc(loss, test_label)
    loss_auc_prc = get_auc_prc(loss, test_label)
    loss_f1s = get_f1_score((valid_diff**2).mean(axis=1),
                            loss,
                            test_label,
                            f1_quantiles=f1_quantiles,
                            )

    return loss, loss_auc_roc, loss_auc_prc, loss_f1s

def get_d_loss(train_diffs,
               valid_diffs,
               test_diffs,
               test_label,
               start_layer_index=0,
               end_layer_index=None,
               gpu_id=-1,
               norm_type=2,
               f1_quantiles=[.99]
               ):
    if end_layer_index is None:
        end_layer_index = len(test_diffs) + 1

    if start_layer_index > len(test_diffs) - 1:
        start_layer_index = len(test_diffs) - 1

    if end_layer_index - start_layer_index < 1:
        end_layer_index = start_layer_index + 1

    #train_diffs = torch.cat([torch.from_numpy(i) for i in train_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    valid_diffs = torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    # |test_diffs| = (batch_size, dim * config.n_layers)

    d_loss = (test_diffs**2).mean(axis=1)
    d_loss_auc_roc = get_auc_roc(d_loss, test_label)
    d_loss_auc_prc = get_auc_prc(d_loss, test_label)
    d_loss_f1s = get_f1_score((valid_diffs**2).mean(axis=1),
                              d_loss,
                              test_label,
                              f1_quantiles=f1_quantiles
                             )

    return d_loss, d_loss_auc_roc, d_loss_auc_prc, d_loss_f1s

def get_d_norm_loss(train_diffs,
                    valid_diffs,
                    test_diffs,
                    test_label,
                    start_layer_index=0,
                    end_layer_index=None,
                    gpu_id=-1,
                    norm_type=2,
                    f1_quantiles=[.99]
                   ):
    if end_layer_index is None:
        end_layer_index = len(test_diffs) + 1

    if start_layer_index > len(test_diffs) - 1:
        start_layer_index = len(test_diffs) - 1

    if end_layer_index - start_layer_index < 1:
        end_layer_index = start_layer_index + 1

    train_diffs = torch.cat([torch.from_numpy(i) for i in train_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    valid_diffs = torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    # |test_diffs| = (batch_size, dim * config.n_layers)

    rotater = Rotater()
    stndzer = Standardizer()

    rotater.fit(train_diffs, gpu_id=gpu_id)
    stndzer.fit(rotater.run(train_diffs, gpu_id=gpu_id))

    valid_rotateds = stndzer.run(rotater.run(valid_diffs, gpu_id=gpu_id))
    test_rotateds = stndzer.run(rotater.run(test_diffs, gpu_id=gpu_id))

    score = get_norm(test_rotateds, norm_type).mean(axis=1)
    auc_roc = get_auc_roc(score, test_label)
    auc_prc = get_auc_prc(score, test_label)
    f1_scores = get_f1_score(get_norm(valid_rotateds, norm_type).mean(axis=1),
                             score,
                             test_label,
                             f1_quantiles=f1_quantiles
                            )

    return score, auc_roc, auc_prc, f1_scores