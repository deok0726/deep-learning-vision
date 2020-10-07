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

    # Raw MNIST
    # import torchvision.datasets as dset
    # import torchvision.transforms as transforms
    # mnist_train = dset.MNIST(args.dataset_root, train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    # mnist_test = dset.MNIST(args.dataset_root, train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
    # train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=args.batch_size, shuffle=True,num_workers=0,drop_last=True)
    # test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=args.batch_size, shuffle=False,num_workers=0,drop_last=True)
    # class DL:
    #     def __init__(self, train_loader, test_loader):
    #         # sample_train_data = mnist_train.__getitem__(np.random.randint(mnist_train.__len__()))[0]
    #         # self.sample_train_data = sample_train_data
    #         # self.sample_train_data = sample_train_data.unsqueeze(0)
    #         self.train_data_loader = train_loader
    #         self.valid_data_loader = test_loader
    # data_loader = DL(train_loader, test_loader)

    # Load Model
    # from models.AE_basic_1 import Model
    # from models.AE_basic_2 import Model
    # from models.AE_RAPP import Model
    from models.CAE_basic_1 import Model
    # from models.CAE_basic_2 import Model
    # from models.CAE_MemAE import Model
    # from models.CAE_ITAE import Model
    # model = Model().to(DEVICE, dtype=torch.float)
    model = Model(n_channels=args.channel_num).to(DEVICE, dtype=torch.float)

    # losses
    losses_dict = dict(
        MSE = torch.nn.MSELoss(reduction='none'),  # squared l2 loss
        # L1 = torch.nn.L1Loss(reduction='none')  # l1 loss
    )
    
    # metrics
    metrics_dict = dict(
        MSE = torch.nn.MSELoss(reduction='none'),
        L1 = torch.nn.L1Loss(reduction='none')
    )
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    # train
    if args.train:
        trainer = Trainer(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        trainer.train()

    # TBD: add tensorboard projector to test data(https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html)
    if args.test:
        metrics_dict['ROC'] = custom_metrics.ROC(args.target_label, args.unique_anomaly)
        tester = Tester(args, data_loader, model, optimizer, losses_dict, metrics_dict, DEVICE)
        tester.test()