import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

# Referred to cifar10_LeNet
class MVTec_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(512, 1024, 5, bias=False, padding=2)
        self.bn2d6 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.conv7 = nn.Conv2d(1024, 1024, 5, bias=False, padding=2)
        self.bn2d7 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.conv8 = nn.Conv2d(1024, 1024, 5, bias=False, padding=2)
        self.bn2d8 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(1024 * 2 * 2, self.rep_dim, bias=False) # metal_nut
        # self.fc1 = nn.Linear(1024 * 3 * 3, self.rep_dim, bias=False) # bottle

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn2d6(x)))
        x = self.conv7(x)
        x = self.pool(F.leaky_relu(self.bn2d7(x)))
        x = self.conv8(x)
        x = self.pool(F.leaky_relu(self.bn2d8(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MVTec_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        
        # bottle == 3*900*900
        # self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        # self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        # self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        # self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        # self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        # self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        # self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        # self.conv6 = nn.Conv2d(512, 1024, 5, bias=False, padding=2)
        # self.bn2d6 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        # self.conv7 = nn.Conv2d(1024, 1024, 5, bias=False, padding=2)
        # self.bn2d7 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        # self.conv8 = nn.Conv2d(1024, 1024, 5, bias=False, padding=2)
        # self.bn2d8 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(1024 * 3 * 3, self.rep_dim, bias=False)

        # metal_nut == 3*700*700
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(512, 1024, 5, bias=False, padding=2)
        self.bn2d6 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.conv7 = nn.Conv2d(1024, 1024, 5, bias=False, padding=2)
        self.bn2d7 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.conv8 = nn.Conv2d(1024, 1024, 5, bias=False, padding=2)
        self.bn2d8 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(1024 * 2 * 2, self.rep_dim, bias=False)

        # Decoder

        # bottle
        # self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 1024, 5, bias=False, padding=2)
        # self.bn2d9 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        # self.deconv2 = nn.ConvTranspose2d(1024, 1024, 5, bias=False, padding=2)
        # self.bn2d10 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        # self.deconv3 = nn.ConvTranspose2d(1024, 512, 5, bias=False, padding=2)
        # self.bn2d11 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        # self.deconv4 = nn.ConvTranspose2d(512, 256, 5, bias=False, padding=3)
        # self.bn2d12 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        # self.deconv5 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=3)
        # self.bn2d13 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.deconv6 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=3)
        # self.bn2d14 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.deconv7 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=3)
        # self.bn2d15 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.deconv8 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        # self.bn2d16 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # self.deconv9 = nn.ConvTranspose2d(16, 3, 5, bias=False, padding=2)

        # metal_nut
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 1024, 5, bias=False, padding=2)
        self.bn2d9 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(1024, 1024, 5, bias=False, padding=2)
        self.bn2d10 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(1024, 512, 5, bias=False, padding=3)
        self.bn2d11 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(512, 256, 5, bias=False, padding=3)
        self.bn2d12 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=4)
        self.bn2d13 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=4)
        self.bn2d14 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv7 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=5)
        self.bn2d15 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv8 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=4)
        self.bn2d16 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv9 = nn.ConvTranspose2d(16, 3, 5, bias=False, padding=4)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn2d6(x)))
        x = self.conv7(x)
        x = self.pool(F.leaky_relu(self.bn2d7(x)))
        x = self.conv8(x)
        x = self.pool(F.leaky_relu(self.bn2d8(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d9(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d10(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d11(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn2d12(x)), scale_factor=2)
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn2d13(x)), scale_factor=2)
        x = self.deconv6(x)
        x = F.interpolate(F.leaky_relu(self.bn2d14(x)), scale_factor=2)
        x = self.deconv7(x)
        x = F.interpolate(F.leaky_relu(self.bn2d15(x)), scale_factor=2)
        x = self.deconv8(x)
        x = F.interpolate(F.leaky_relu(self.bn2d16(x)), scale_factor=2)
        x = self.deconv9(x)
        x = torch.sigmoid(x)
        return x
