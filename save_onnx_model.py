import io
import numpy as np
import random
import os

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import argument_parser as parser

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
    elif args.model_name=='MemAE':
        from models.CAE_MemAE import Model
        model = Model(args.channel_num, args.input_height, args.input_width, args.memory_dimension).to(DEVICE, dtype=torch.float)
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

def _restore_checkpoint(CHECKPOINT_SAVE_DIR, model, optimizer):
    ckpts_list = os.listdir(CHECKPOINT_SAVE_DIR)
    if ckpts_list:
        last_epoch = sorted(list(map(int, [epoch.split('_')[-1].split('.')[0] for epoch in ckpts_list])))[-1]
        print('Restore Checkpoint epoch ', last_epoch)
        states = torch.load(os.path.join(CHECKPOINT_SAVE_DIR, 'epoch_{}.tar'.format(last_epoch)))
        # epoch_idx = states['epoch']
        model.load_state_dict(states['model_state_dict'])
        optimizer.load_state_dict(states['optimizer_state_dict'])
    else:
        print('No checkpoints to restore')

def automkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+' generated')
    else:
        print(path+' exists')


if __name__ == "__main__":
    args = parser.get_args()
    args.input_height, args.input_width = 28, 28
    CHECKPOINT_SAVE_DIR = os.path.join(os.path.join(args.checkpoint_dir, args.model_name), args.exp_name)
    # '/hd/checkpoints/MemAE/MNIST_Data_MemAE_Model_target_1/'
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
    
    model = load_model(args)
    optimizer, lr_scheduler = load_optimizer_with_lr_scheduler(args)
    _restore_checkpoint(CHECKPOINT_SAVE_DIR, model, optimizer)
    model.eval()
    x = torch.randn(1, 1, 28, 28, requires_grad=True).to(DEVICE)
    torch_out = model(x)
    onnx_model_save_path = os.path.join("/hd/onnx_model/", args.model_name)
    automkdir(onnx_model_save_path)
    onnx_model_save_path = os.path.join(onnx_model_save_path, args.exp_name)
    automkdir(onnx_model_save_path)
    torch.onnx.export(model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  os.path.join(onnx_model_save_path, "/hd/anoamly_detection_model.onnx"),   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

