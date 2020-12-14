from codes import mvtecad
from codes import gms
from codes import daejoo
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores
import sys


__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(x, enc, K, S):
    x = NHWC2NCHW(x)
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps, dataset):
    if dataset == 'mvtec':
        auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

        anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
        auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
        return auroc_det, auroc_seg

    elif dataset == 'gms':
        anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
        print(anomaly_scores)
        result = gms._classification_report(anomaly_scores, obj, 3.3, target_label=1)
        print(result)
        auroc_det = gms.detection_auroc(obj, anomaly_scores)
        return auroc_det
    
    elif dataset == 'daejoo':
        anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
        print(anomaly_scores)
        result = daejoo._classification_report(anomaly_scores, obj, 3.3, target_label=1)
        print(result)
        auroc_det = daejoo.detection_auroc(obj, anomaly_scores)
        if auroc_det == None:
            return 0
        else:
            return auroc_det


#########################

def eval_encoder_NN_multiK(enc, obj, dataset):
    if dataset == 'mvtec':
        x_tr = mvtecad.get_x_standardized(obj, mode='train')
        x_te = mvtecad.get_x_standardized(obj, mode='test')

        embs64_tr = infer(x_tr, enc, K=64, S=16)
        embs64_te = infer(x_te, enc, K=64, S=16)

        x_tr = mvtecad.get_x_standardized(obj, mode='train')
        x_te = mvtecad.get_x_standardized(obj, mode='test')

        embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
        embs32_te = infer(x_te, enc.enc, K=32, S=4)

        embs64 = embs64_tr, embs64_te
        embs32 = embs32_tr, embs32_te
    
    elif dataset == 'gms':
        x_tr = gms.get_x_standardized(obj, mode='train')
        x_te = gms.get_x_standardized(obj, mode='test')

        embs64_tr = infer(x_tr, enc, K=64, S=16)
        embs64_te = infer(x_te, enc, K=64, S=16)

        x_tr = gms.get_x_standardized(obj, mode='train')
        x_te = gms.get_x_standardized(obj, mode='test')

        embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
        embs32_te = infer(x_te, enc.enc, K=32, S=4)

        embs64 = embs64_tr, embs64_te
        embs32 = embs32_tr, embs32_te
    
    elif dataset == 'daejoo':
        x_tr = daejoo.get_x_standardized(obj, mode='train')
        x_te = daejoo.get_x_standardized(obj, mode='test')

        embs64_tr = infer(x_tr, enc, K=64, S=16)
        embs64_te = infer(x_te, enc, K=64, S=16)

        x_tr = daejoo.get_x_standardized(obj, mode='train')
        x_te = daejoo.get_x_standardized(obj, mode='test')

        embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
        embs32_te = infer(x_te, enc.enc, K=32, S=4)

        embs64 = embs64_tr, embs64_te
        embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(obj, embs64, embs32, dataset)


def eval_embeddings_NN_multiK(obj, embs64, embs32, dataset, NN=1):
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    if dataset == 'mvtec':
        det_64, seg_64 = assess_anomaly_maps(obj, maps_64, dataset)
    elif dataset == 'gms':
        det_64 = assess_anomaly_maps(obj, maps_64, dataset)
        seg_64 = 0
    elif dataset == 'daejoo':
        det_64 = assess_anomaly_maps(obj, maps_64, dataset)
        seg_64 = 0

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    if dataset == 'mvtec':
        det_32, seg_32 = assess_anomaly_maps(obj, maps_32, dataset)
    elif dataset == 'gms':
        det_32 = assess_anomaly_maps(obj, maps_32, dataset)
        seg_32 = 0
    elif dataset == 'daejoo':
        det_32 = assess_anomaly_maps(obj, maps_32, dataset)
        seg_32 = 0

    maps_sum = maps_64 + maps_32
    if dataset == 'mvtec':
        det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum, dataset)
    elif dataset == 'gms':
        det_sum = assess_anomaly_maps(obj, maps_sum, dataset)
        seg_sum = 0
    elif dataset == 'daejoo':
        det_sum = assess_anomaly_maps(obj, maps_sum, dataset)
        seg_sum = 0

    maps_mult = maps_64 * maps_32
    if dataset == 'mvtec':
        det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult, dataset)
    elif dataset == 'gms':
        det_mult = assess_anomaly_maps(obj, maps_mult, dataset)
        seg_mult = 0
    elif dataset == 'daejoo':
        det_mult = assess_anomaly_maps(obj, maps_mult, dataset)
        seg_mult = 0

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
