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

if __name__ == '__main__':
    # get arguments
    args = parser.get_args()
    print('running config :', args)

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    USE_CUDA = torch.cuda.is_available()
    
    # Reproducibility
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
    from models.AE_tmp import Model
    model = Model(n_channels=1).to(DEVICE, dtype=torch.float)

    # losses
    losses_dict = dict(
        MSE = torch.nn.MSELoss(reduction='none'),  # squared l2 loss
        L1 = torch.nn.L1Loss(reduction='none')  # l1 loss
    )
    
    # metrics
    metrics_dict = dict(
        MSE = torch.nn.MSELoss(reduction='none'),
        L1 = torch.nn.L1Loss(reduction='none')
    )
    
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)

    # train
    if args.train:
        trainer = Trainer(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        trainer.train()

    # TBD: add tensorboard projector to test data(https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html)
    metrics_dict['ROC'] = custom_metrics.ROC(args.target_label)
    if args.test:
        tester = Tester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        tester.test()