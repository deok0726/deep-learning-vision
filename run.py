import torch
import torchvision
import numpy as np
import random
import os
import time, datetime
import argument_parser as parser
from main.trainers.trainer import Trainer
from main.testers.tester import Tester
from data_loader import DataLoader
from modules import custom_metrics
from modules.utils import AverageMeter


def load_loss(losses_name):
    losses_dict = {}
    if 'MSE' in losses_name:
        losses_dict['MSE'] = torch.nn.MSELoss(reduction='none') # l2 loss
    if 'L1' in losses_name:
        losses_dict['L1'] = torch.nn.L1Loss(reduction='none') # l1 loss
    return losses_dict

def load_model(args):
    if args.model_name=='ARNet':
        from models.CAE_ARNet import Model
        if args.dataset_name=='MvTec' or args.dataset_name=='GMS':
            if args.graying:
                model = Model(in_channels=1, out_channels=3, bilinear=False).to(DEVICE, dtype=torch.float)
            else:
                model = Model(in_channels=3, out_channels=3, bilinear=False).to(DEVICE, dtype=torch.float)
        else:
            model = Model(in_channels=1, out_channels=1, bilinear=False).to(DEVICE, dtype=torch.float)
        # x = torch.rand(1,1,300,300).to(DEVICE, dtype=torch.float)
        # _children = model.children()
        # for layer in _children:
        #     print(layer)
        #     x = layer(x)
        #     print(x.shape)
        # _modules = model.modules()
        # for layer in _modules:
        #     print(layer)
        #     x = layer(x)
        #     print(x.shape)
        # print('test')
    elif args.model_name=='MemAE':
        from models.CAE_MemAE import Model
        model = Model(args.channel_num, args.input_height, args.input_width).to(DEVICE, dtype=torch.float)
        # model = Model(n_channels=args.channel_num, mem_dim = 100).to(DEVICE, dtype=torch.float)
    elif args.model_name=='MvTec':
        from models.CAE_MvTec import Model
        model = Model(n_channels=args.channel_num).to(DEVICE, dtype=torch.float)
    elif args.model_name=='RaPP':
        from models.AE_RaPP import Model
        model = Model(n_channels=args.channel_num).to(DEVICE, dtype=torch.float)
    elif args.model_name=='CAE':
        from models.CAE_basic_2 import Model
        model = Model(n_channels=args.channel_num).to(DEVICE, dtype=torch.float)
    else:
        raise NotImplementedError
    return model

def load_optimizer_with_lr_scheduler(args):
    optimizer = None
    lr_scheduler = None
    if args.model_name=='ARNet':
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
        if args.learning_rate_decay:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25*8, gamma=args.learning_rate_decay_ratio)
    elif args.model_name=='MemAE':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    elif args.model_name=='MvTec':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
        if args.learning_rate_decay:
            multiplier = (args.end_learning_rate / args.learning_rate) ** (1.0/(float(args.num_epoch)-1))
            lr_scheduler_function = lambda epoch: multiplier
            lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_scheduler_function)
    elif args.model_name=='RaPP':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    return optimizer, lr_scheduler

def load_trainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE):
    if args.model_name=='ARNet':
        from main.trainers.ARNet_trainer import ARNetTrainer
        trainer = ARNetTrainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
    elif args.model_name=='MemAE':
        from main.trainers.MemAE_trainer import MemAETrainer
        trainer = MemAETrainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
    elif args.model_name=='MvTec':
        from main.trainers.MvTec_trainer import MvTecTrainer
        trainer = MvTecTrainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
    else:
        trainer = Trainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
    return trainer

def load_tester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE):
    if args.model_name=='ARNet':
        from main.testers.ARNet_tester import ARNetTester
        tester = ARNetTester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
    elif args.model_name=='MemAE':
        from main.testers.MemAE_tester import MemAETester
        tester = MemAETester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
    elif args.model_name=='MvTec':
        from main.testers.MvTec_tester import MvTecTester
        tester = MvTecTester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
    else:
        tester = Tester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
    return tester

if __name__ == '__main__':
    # get arguments
    args = parser.get_args()
    print('running config :', args)

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    USE_CUDA = torch.cuda.is_available()
    
    # Reproducibility
    if args.reproducibility:
        random_seed = args.random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if(USE_CUDA):
            torch.cuda.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
            
            # Deterministic operation(may have a negative single-run performance impact, depending on the composition of your model.)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    
    # Select Device
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using Device: ",DEVICE)
    # TBD: Multi-GPU

    # Load Data
    data_loader = DataLoader(args)

    # Load Model
    model = load_model(args)
    

    # losses
    losses_dict = load_loss([
        # 'MSE', 
        # 'L1'
        ])

    # metrics
    metrics_dict = dict(
        MSE = torch.nn.MSELoss(reduction='none'),
        L1 = torch.nn.L1Loss(reduction='none')
    )
    
    # optimizer
    optimizer, lr_scheduler = load_optimizer_with_lr_scheduler(args)

    # train
    if args.train:
        trainer = load_trainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
        trainer.train()

    # TBD: add tensorboard projector to test data(https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html)
    if args.test:
        metrics_dict['AUROC'] = custom_metrics.AUROC(args.target_label, args.unique_anomaly)
        tester = load_tester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        tester.test()