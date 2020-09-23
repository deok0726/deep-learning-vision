import torch
import torchvision
import numpy as np
import random
import os
import time, datetime
import argument_parser as parser
from utils.utils import AverageMeter
from main.trainers.trainer import Trainer
from data_loader import DataLoader

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
    losses = dict(
        mse = torch.nn.MSELoss()  # squared l2 loss
        )
    
    # metrics
    metrics = dict(
        mse = torch.nn.MSELoss()
    )
    
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)

    # train
    if args.train:
        trainer = Trainer(args, data_loader, model, losses, optimizer, metrics, DEVICE)
        trainer.train()