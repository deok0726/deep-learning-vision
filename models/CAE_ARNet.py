import torch
import torch.nn as nn
import torch.nn.functional as F


# Refer to https://github.com/FishSmile-syx/ITAE-Pytorch-Anomaly_Detection/blob/master/model/ITAE.py
# Refer to https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # or leakyRelu
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # or leakyRelu
            # nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Model(nn.Module):
    """ ITAE model """

    def __init__(self, in_channels=1, out_channels=3, bilinear=True):
        super(Model, self).__init__()
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64)
        # seems that the structure in paper does not contain 'tanh'
        # self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)# Unet use 1*1conv to be out_conv
        self.tanh = torch.tanh
        self.rec_criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x3_3 = self.down3(x2_3)
        x4_3 = self.down4(x3_3)
        x = self.up1(x4_3, x3_3)
        x = self.up2(x, x2_3)
        x = self.up3(x, x1_3)
        x = self.up4(x, x0_2)
        x = self.out_conv(x)
        out = self.tanh(x)
        return out
    
    def get_losses_name(self):
        return ['rec_loss']

    def get_losses(self, args_for_losses):
        rec_loss = self._get_recon_loss(args_for_losses['x'], args_for_losses['y'])
        return {'rec_loss': rec_loss}

    def _get_recon_loss(self, x, y):
        rec_loss = self.rec_criterion(y, x)
        return rec_loss