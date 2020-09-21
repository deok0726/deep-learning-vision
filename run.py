import torch
import torchvision
import os
# from data_tools.utils import AverageMeter
# from utils import AverageMeter
# from torch_utils import AverageMeter
from utils.utils import AverageMeter
import time, datetime

import argument_parser as parser
from main import trainer

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using Device: ",DEVICE)

    args = parser.get_args()
    print('running config :', args)

    # DATA
    ## download mnist
    train_data = torchvision.datasets.MNIST(root='data_tmp/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data  = torchvision.datasets.MNIST(root='data_tmp/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    print(train_data[0][0].shape)
    ## DataLoader tmp
    train_data_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True)
    ## mini batch
    # example_mini_batch_img, example_mini_batch_label  = next(iter(train_data_loader))
    # print(example_mini_batch_img.shape)

    # MODEL
    # from models.AE import Autoencoder
    # model = Autoencoder().to(DEVICE)
    from models.AE_tmp import Model
    model = Model().to(DEVICE)

    # loss & optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
    loss = torch.nn.MSELoss()   # squared l2 loss

    # experiment & train general
    for epoch_idx in range(0,args.num_epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train() # nn.Module

        end_time = time.time()
        for batch_idx, (batch_imgs, batch_label) in enumerate(train_data_loader):
            print('start')
            print(batch_idx,batch_imgs.shape)
            data_time.update(time.time() - end_time)
            curr_time = datetime.datetime.now()
            curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
            # train detail step 
            batch_imgs = batch_imgs.cuda()
            output_imgs = model(batch_imgs)
            cost = loss(output_imgs, batch_imgs)
            model.zero_grad()
            cost.backward()
            optimizer.step()
            losses.update(cost.item(),batch_imgs.size(0))
            #
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            print("current time :\t",curr_time)
            print("epoch index :\t",epoch_idx)
            print("batch index :\t",batch_idx+1)
            print("learning rate :\t",optimizer.param_groups[0]['lr'])
            print("batch_time :\t",batch_time.avg)
            print("data_time :\t",data_time.avg)
            print("loss :\t\t", losses.avg)
        if (epoch_idx+1) % 1 == 0:
            states = {
                'epoch': epoch_idx + 1,
                'state_dict': model.state_dict()
                }
            torch.save(states, os.path.join(args.save_dir, 'tmp_model_epoch{}.pth'.format(epoch_idx + 1)))
                