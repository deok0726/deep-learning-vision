from torch import nn


# Refer to https://wjddyd66.github.io/pytorch/Pytorch-AutoEncoder/
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Linear(28*28,20)
        self.decoder = nn.Linear(20,28*28)

    def forward(self, x):
        encoded = self.encoder(x.view(x.shape[0], -1))
        return self.decoder(encoded).view(x.shape)