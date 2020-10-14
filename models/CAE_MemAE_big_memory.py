import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from collections import namedtuple

# Refer to https://github.com/h19920918/memae/blob/master/model.py
class Model(nn.Module):
    def __init__(self, n_channels, mem_dim=100, shrink_thres=0.0025):
        super(Model, self).__init__()
        self.encoder = Encoder(n_channels)
        self.decoder = Decoder(n_channels)
        self.memory_module = MemoryModule(mem_dim=100, fea_dim=64, shrink_thres=shrink_thres)
        self.rec_criterion = nn.MSELoss(reduction='none')
        self.return_values = namedtuple("return_values", 'output mem_weight')

    def forward(self, x):
        z = self.encoder(x)
        f = self.memory_module(z)
        mem_weight = f['mem_weight']
        # output = self.decoder(z)
        output = self.decoder(f['output'])
        return self.return_values(output, mem_weight)
        # return output
    
    def get_losses_name(self):
        return ['entropy_loss', 'rec_loss']

    def get_losses(self, args_for_losses):
        entropy_loss = self._get_entropy_loss(args_for_losses['batch_size'], args_for_losses['weight'], args_for_losses['entropy_loss_coef'])
        rec_loss = self._get_recon_loss(args_for_losses['x'], args_for_losses['y'])
        return {'entropy_loss': entropy_loss, 'rec_loss': rec_loss}

    def _get_entropy_loss(self, batch_size, weight, entropy_loss_coef):
        mask = (weight == 0).float()
        temp = torch.numel(weight[mask==1])/torch.numel(weight)
        maksed_weight = weight + mask
        entropy_loss = -weight * torch.log(maksed_weight)
        entropy_loss *= entropy_loss_coef
        return entropy_loss
    
    def _get_recon_loss(self, x, y):
        rec_loss = self.rec_criterion(y, x)
        return rec_loss

class Encoder(nn.Module):
    def __init__(self, image_channel_num):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channel_num, 16, kernel_size=1,stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, image_channel_num):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3,stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2,stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, image_channel_num, kernel_size=2,stride=2, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)

        return x

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        mem_weight = F.linear(x, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        mem_weight = F.softmax(mem_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            mem_weight = hard_shrink_relu(mem_weight, lambd=self.shrink_thres)
            mem_weight = F.normalize(mem_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(mem_weight, mem_trans)  # mem_weight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        # return output
        return {'output': output, 'mem_weight': mem_weight}  # output, mem_weight

# Refer to https://github.com/donggong1/memae-anomaly-detection/blob/master/models/memory_module.py
# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        mem_weight = y_and['mem_weight']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            mem_weight = mem_weight.view(s[0], s[2], self.mem_dim)
            mem_weight = mem_weight.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            mem_weight = mem_weight.view(s[0], s[2], s[3], self.mem_dim)
            mem_weight = mem_weight.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            mem_weight = mem_weight.view(s[0], s[2], s[3], s[4], self.mem_dim)
            mem_weight = mem_weight.permute(0, 4, 1, 2, 3)
        else:
            y = x
            mem_weight = mem_weight
            print('wrong feature map size')
        return {'output': y, 'mem_weight': mem_weight}

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    output = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
    return output
