import os
import sys
import argparse
import torch
from codes import mvtecad
from codes import gms
from codes import daejoo
from codes import etc
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *
from tensorboardX import SummaryWriter

# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='hazelnut', type=str)
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--dataset', default='etc', type=str)
parser.add_argument('--data_path', default='/hd/', type=str)

def train():
    args = parser.parse_args()
    obj = args.obj
    D = args.D
    lr = args.lr
    dataset = args.dataset
    data_path = args.data_path

    print(args)

    if dataset == 'mvtec':
        writer = SummaryWriter('runs/mvtec_experiment')
    elif dataset == 'gms':
        writer = SummaryWriter('runs/gms_experiment/')
    elif dataset == 'daejoo':
        writer = SummaryWriter('runs/daejoo_experiment/')
    elif dataset == 'etc':
        writer = SummaryWriter('runs/etc_experiment/')

    with task('Networks'):
        enc = EncoderHier(64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        if dataset == 'mvtec':
            train_x = mvtecad.get_x_standardized(obj, mode='train')
            train_x = NHWC2NCHW(train_x)
            # valid_x = mvtecad.get_x_standardized(obj, mode='valid')
            # valid_x = NHWC2NCHW(train_x)
        
        elif dataset == 'gms':
            train_x = gms.get_x_standardized(obj, mode='train')
            train_x = NHWC2NCHW(train_x)
            valid_x = gms.get_x_standardized(obj, mode='valid')
            valid_x = NHWC2NCHW(valid_x)
        
        elif dataset == 'daejoo':
            train_x = daejoo.get_x_standardized(obj, mode='train')
            train_x = NHWC2NCHW(train_x)
            valid_x = daejoo.get_x_standardized(obj, mode='valid')
            valid_x = NHWC2NCHW(valid_x)
        
        elif dataset == 'etc':
            etc.set_root_path(data_path)
            train_x = etc.get_x_standardized(obj, mode='train')
            train_x = NHWC2NCHW(train_x)
            valid_x = etc.get_x_standardized(obj, mode='valid')
            valid_x = NHWC2NCHW(valid_x)

        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        datasets_val = dict()
        datasets_val[f'pos_64'] = PositionDataset(valid_x, K=64, repeat=rep)
        datasets_val[f'pos_32'] = PositionDataset(valid_x, K=32, repeat=rep)
        datasets_val[f'svdd_64'] = SVDD_Dataset(valid_x, K=64, repeat=rep)
        datasets_val[f'svdd_32'] = SVDD_Dataset(valid_x, K=32, repeat=rep)

        dataset_tr = DictionaryConcatDataset(datasets)
        dataset_val = DictionaryConcatDataset(datasets_val)
        loader = DataLoader(dataset_tr, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        loader_val = DataLoader(dataset_val, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training')
    loss = 0
    loss_val = 0
    for i_epoch in range(args.epochs):
        print(i_epoch+1, "/", args.epochs)
        for module in modules:
            module.train()

        for d in loader:
            d = to_device(d, 'cuda', non_blocking=True)
            opt.zero_grad()

            loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
            loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
            loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
            loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

            loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

            loss.backward()
            opt.step()
            
        for module in modules:
            module.eval()
        
        for vd in loader_val:
            vd = to_device(vd, 'cuda', non_blocking=True)
            
            with torch.no_grad():
                loss_pos_64 = PositionClassifier.infer(cls_64, enc, vd['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, vd['pos_32'])
                loss_svdd_64 = SVDD_Dataset.infer(enc, vd['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, vd['svdd_32'])

            loss_val = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)
    
        if dataset == 'mvtec':
            writer.add_scalar('training_loss', loss, i_epoch)
        elif dataset == 'gms':
            writer.add_scalars('training_validation', {'training_loss' : loss, 'validation_loss' : loss_val}, i_epoch)
        elif dataset == 'daejoo':
            writer.add_scalars('training_validation', {'training_loss' : loss, 'validation_loss' : loss_val}, i_epoch)
        elif dataset == 'etc':
            writer.add_scalars('training_validation', {'training_loss' : loss, 'validation_loss' : loss_val}, i_epoch)
        
        aurocs = eval_encoder_NN_multiK(enc, obj, dataset)
        log_result(obj, aurocs)
        enc.save(obj)
    writer.close()


def log_result(obj, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |mult| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(1)
    print('GPU Number is :', torch.cuda.current_device())
    train()
