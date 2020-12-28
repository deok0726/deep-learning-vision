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
        assert input_height == self.calDim(input_height), 'input dimension output dimension mismatch'
        self.conv_channel_num = 16
        # self.conv_channel_num = 4
        self.encoder = Encoder(n_channels, self.conv_channel_num)
        self.decoder = Decoder(n_channels, self.conv_channel_num)
        temp_encoder_output = self.encoder.forward(torch.rand(1, n_channels, input_height, input_width))
        self.input_h = temp_encoder_output.shape[-2]
        self.input_w = temp_encoder_output.shape[-1]
        self.rec_criterion = nn.MSELoss(reduction='none')
        self.return_values = namedtuple("return_values", 'output mem_weight embedding')

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    
    def get_losses_name(self):
        return ['rec_loss']

    def get_losses(self, args_for_losses):
        rec_loss = self._get_recon_loss(args_for_losses['x'], args_for_losses['y'])
        return {'rec_loss': rec_loss}

    def _get_recon_loss(self, x, y):
        rec_loss = self.rec_criterion(y, x)
        return rec_loss
    
    def calDim(self, crop_size):
        x1 = math.floor((crop_size+2-(4-1)-1)/2+1)
        x2 = math.floor((x1+2-(4-1)-1)/2+1)
        x3 = math.floor((x2+2-(4-1)-1)/2+1)
        y1 = math.floor((x3+2-(3-1)-1)/1+1)*2
        y2 = math.floor((y1+2-(3-1)-1)/1+1)*2
        y3 = math.floor((y2+2-(3-1)-1)/1+1)*2
        print(crop_size, x1, x2, x3, y1, y2, y3)
        return y3

class Encoder(nn.Module):
    def __init__(self, image_channel_num, conv_channel_num):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channel_num, conv_channel_num, kernel_size=4,stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channel_num)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_channel_num, conv_channel_num*2, kernel_size=4,stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channel_num*2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(conv_channel_num*2, conv_channel_num*4, kernel_size=4,stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channel_num*4)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, image_channel_num, conv_channel_num):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(conv_channel_num*4, conv_channel_num*2*(2**2), kernel_size=3, stride=1, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.bn1 = nn.BatchNorm2d(conv_channel_num*2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_channel_num*2, conv_channel_num*(2**2), kernel_size=3,stride=1, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.bn2 = nn.BatchNorm2d(conv_channel_num)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(conv_channel_num, image_channel_num*(2**2), kernel_size=3,stride=1, padding=1)
        self.ps3 = nn.PixelShuffle(2)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ps1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.ps2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.ps3(x)

        return x