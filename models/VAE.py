import torch
import torch.nn as nn
import torch.nn.functional as F


# Refer to https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2).cuda()

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
    """ ITAE -> Variational AE model """

    def __init__(self, n_channels=3, latent_dim=10, bilinear=True):
        super(Model, self).__init__()

        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        self.decode_input = nn.Linear(self.latent_dim, 512)
        
        self.in_conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        # seems that the structure in paper does not contain 'tanh'
        self.out_conv = nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv


    def get_mu_var(self, encoded):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        # """
        # flattened = encoded
        flattened = torch.flatten(encoded, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x3_3 = self.down3(x2_3)
        x4_3 = self.down4(x3_3)

        self.mu, self.log_var = self.get_mu_var(x4_3)
        z = self.reparameterize(self.mu, self.log_var)
        decoder_input = self.decode_input(z)
        decoder_input = decoder_input.view(-1, 512, 1, 1)

        x = self.up1(decoder_input, x3_3)
        # x = self.up1(x4_3, x3_3)
        x = self.up2(x, x2_3)
        x = self.up3(x, x1_3)
        x = self.up4(x, x0_2)
        out = torch.tanh(self.out_conv(x))
        return out

# class kld_custom(nn.Module):
    def set_kld_loss(self,recons_loss):
    # def set_kld_loss(self,model_output, model_input):
        kld_weight = 0.0005
        # recons_loss = torch.nn.MSELoss(reduction='none')(model_output, model_input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu **2 - self.log_var.exp(), dim = 1), dim = 0)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu **2 - self.log_var.exp(), dim = 1), dim = 1)
        total_loss = recons_loss + kld_weight * kld_loss
        return total_loss, kld_loss