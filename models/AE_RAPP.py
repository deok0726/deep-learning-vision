from torch import nn

class Model(nn.Module):
    def __init__(self, n_channels):
        super(Model, self).__init__()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, (28 * 28)-(76*1)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*1)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*1), (28 * 28)-(76*2)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*2)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*2), (28 * 28)-(76*3)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*3)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*3), (28 * 28)-(76*4)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*4)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*4), (28 * 28)-(76*5)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*5)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*5), (28 * 28)-(76*6)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*6)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*6), (28 * 28)-(76*7)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*7)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*7), (28 * 28)-(76*8)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*8)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*8), (28 * 28)-(76*9)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*9)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*9), 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, (28 * 28)-(76*9)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*9)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*9), (28 * 28)-(76*8)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*8)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*8), (28 * 28)-(76*7)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*7)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*7), (28 * 28)-(76*6)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*6)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*6), (28 * 28)-(76*5)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*5)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*5), (28 * 28)-(76*4)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*4)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*4), (28 * 28)-(76*3)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*3)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*3), (28 * 28)-(76*2)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*2)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*2), (28 * 28)-(76*1)),
            nn.ReLU(True),
            nn.BatchNorm1d((28 * 28)-(76*1)),
            nn.Dropout(p=0),
            nn.Linear((28 * 28)-(76*1), 28 * 28)
            )
        

    def forward(self, x):
        if self.n_channels == 1:
            encoded = self.encoder(x.view(-1, x.shape[-1]*x.shape[-2]))
        elif self.n_channels == 3:
            encoded = self.encoder(x.view(-1, x.shape[-1]*x.shape[-2]*x.shape[-3]))
        return self.decoder(encoded).view(x.shape)
