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
    # from models.AE_basic_1 import Model
    # from models.AE_basic_2 import Model
    # from models.AE_RAPP import Model
    # from models.CAE_basic_1 import Model
    # from models.CAE_basic_2 import Model
    # from models.CAE_MvTec import Model
    # from models.CAE_MemAE import Model
    # model = Model(n_channels=args.channel_num, mem_dim = 100).to(DEVICE, dtype=torch.float)
    # from models.CAE_MemAE_big_memory import Model
    from models.CAE_ARNet import Model
    # model = Model().to(DEVICE, dtype=torch.float)
    # model = Model(n_channels=args.channel_num).to(DEVICE, dtype=torch.float)
    model = Model(n_channels=args.channel_num, bilinear=False).to(DEVICE, dtype=torch.float)
    

    # losses
    losses_dict = dict(
        # MSE = torch.nn.MSELoss(reduction='none'),  # squared l2 loss
        # L1 = torch.nn.L1Loss(reduction='none')  # l1 loss
    )
    
    # metrics
    metrics_dict = dict(
        MSE = torch.nn.MSELoss(reduction='none'),
        L1 = torch.nn.L1Loss(reduction='none')
    )
    
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.learning_rate_decay:
        # lambda lr
        # lrs = np.linspace(args.learning_rate, args.end_learning_rate, args.num_epoch)
        # lr_scheduler_function = lambda epoch: lrs[epoch]
        # def lr_scheduler_function(epoch):
        #     print("epoch: ", epoch)
        #     print("decayed learning rate: ", lrs[epoch])
        #     return lrs[epoch]
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_function, last_epoch=0)
        # learning rate -> end_lr
        # multiplier = (args.end_learning_rate / args.learning_rate) ** (1.0/(float(args.num_epoch)-1))
        # lr_scheduler_function = lambda epoch: multiplier
        # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_scheduler_function)
        # step lr
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=args.learning_rate_decay_ratio)
    else:
        lr_scheduler = None

    # train
    if args.train:
        if args.model_name == 'MemAE':
            from main.trainers.MemAE_trainer import MemAETrainer
            trainer = MemAETrainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
        elif args.model_name == 'ARNet':
            from main.trainers.ARNet_trainer import ARNetTrainer
            trainer = ARNetTrainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
        else:
            trainer = Trainer(args, data_loader, model, optimizer, lr_scheduler, losses_dict, metrics_dict, DEVICE)
        trainer.train()

    # TBD: add tensorboard projector to test data(https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html)
    if args.test:
        metrics_dict['ROC'] = custom_metrics.ROC(args.target_label, args.unique_anomaly)
        if args.model_name == 'MemAE':
            from main.testers.MemAE_tester import MemAETester
            tester = MemAETester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        if args.model_name == 'ARNet':
            from main.testers.ARNet_tester import ARNetTester
            tester = ARNetTester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        else:
            tester = Tester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        tester.test()