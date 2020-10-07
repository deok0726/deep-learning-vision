from torch import nn


# Refer to https://github.com/h19920918/memae/blob/master/model.py
class Model(nn.Module):
    def __init__(self, n_channels):
        super(Model, self).__init__()
        self.encoder = Encoder(n_channels)
        self.decoder = Decoder(n_channels)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

class Encoder(nn.Module):
    def __init__(self, image_channel_num):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channel_num, 16, kernel_size=1,stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 100, kernel_size=3,stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(100)
    
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
        self.deconv1 = nn.ConvTranspose2d(100, 32, kernel_size=3,stride=2, padding=1, output_padding=1)
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