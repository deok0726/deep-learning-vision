from torch import nn
from torch import tanh

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
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(image_channel_num, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(32, 100, kernel_size=8, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv3(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv4(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv5(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv6(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv7(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv8(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.conv9(x)
        return x

class Decoder(nn.Module):
    def __init__(self, image_channel_num):
        super(Decoder, self).__init__()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(100, 32, kernel_size=8, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv9 = nn.ConvTranspose2d(32, image_channel_num, kernel_size=4, stride=2, padding=1)        
    
    def forward(self, x):
        x = self.deconv1(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv2(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv3(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv4(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv5(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv6(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv7(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv8(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv8(x)
        # x = self.leakyRelu(x)
        x = self.relu(x)
        x = self.deconv9(x)
        x = tanh(x)
        return x