import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from collections import namedtuple

# Refer to https://github.com/h19920918/memae/blob/master/model.py
class Model(nn.Module):
    def __init__(self, n_channels, input_height, input_width, mem_dim=100):
        super(Model, self).__init__()
        self.conv_channel_num = 16
        self.encoder = Encoder(n_channels, self.conv_channel_num)
        self.decoder = Decoder(n_channels, self.conv_channel_num)
        temp_encoder_output = self.encoder.forward(torch.rand(1, n_channels, input_height, input_width))
        self.input_h = temp_encoder_output.shape[-2]
        self.input_w = temp_encoder_output.shape[-1]
        self.memory_module = MemoryModule(mem_dim=mem_dim, fea_dim=self.conv_channel_num*4, input_h=self.input_h, input_w=self.input_w)
        # self.memory_module = MemoryModule(mem_dim=mem_dim, fea_dim=self.conv_channel_num*4, input_h=self.input_h, input_w=self.input_w)
        self.rec_criterion = nn.MSELoss(reduction='none')
        self.return_values = namedtuple("return_values", 'output mem_weight')

    def forward(self, x):
        z = self.encoder(x)
        f = self.memory_module(z)        
        mem_weight = f['mem_weight']
        # output = self.decoder(z)
        output = self.decoder(f['output'].view(-1, self.conv_channel_num*4, self.input_h, self.input_w))
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
    def __init__(self, image_channel_num, conv_channel_num):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channel_num, conv_channel_num, kernel_size=1,stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channel_num)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_channel_num, conv_channel_num*2, kernel_size=3,stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channel_num*2)
        self.conv3 = nn.Conv2d(conv_channel_num*2, conv_channel_num*4, kernel_size=3,stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channel_num*4)
    
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
    def __init__(self, image_channel_num, conv_channel_num):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(conv_channel_num*4, conv_channel_num*2, kernel_size=3,stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channel_num*2)
        self.deconv2 = nn.ConvTranspose2d(conv_channel_num*2, conv_channel_num, kernel_size=2,stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channel_num)
        self.deconv3 = nn.ConvTranspose2d(conv_channel_num, image_channel_num, kernel_size=2,stride=2, padding=1)
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
    def __init__(self, mem_dim, fea_dim, input_h=4, input_w=4):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim * input_h * input_w))  # M x C(fea_dim * h * w)
        self.bias = None
        self.shrink_thres = 1 / self.weight.shape[0]
        self.cosine_similarity = nn.CosineSimilarity(dim=2,)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        batch = x.shape[0]
        ex_mem = self.weight.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = x.unsqueeze(1).repeat(1, self.mem_dim, 1)
        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)
        if(self.shrink_thres>0):
            mem_weight = (self.relu(mem_weight - self.shrink_thres) * mem_weight) / (torch.abs(mem_weight - self.shrink_thres) + 1e-12)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1).unsqueeze(1).expand(batch, self.mem_dim)
        output = torch.mm(mem_weight, self.weight)
        return {'output': output, 'mem_weight': mem_weight}  # output, mem_weight

# Refer to https://github.com/h19920918/memae/blob/master/model.py
class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, input_h, input_w):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, input_h, input_w)

    def forward(self, x):
        batch = x.data.shape[0]
        x = x.view(batch, -1)
        y_and = self.memory(x)
        y = y_and['output']
        mem_weight = y_and['mem_weight']
        return {'output': y, 'mem_weight': mem_weight}

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    output = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
    return output
