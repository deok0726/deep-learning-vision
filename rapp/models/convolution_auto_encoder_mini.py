#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import torch
from torch import nn
from models import AbstractModel
from modules import Loss



class ConvolutionAutoEncoder(AbstractModel):

    def __init__(
            self,
            input_channel,
            btl_size=100,
            encoder_and_opt=None,
            decoder_and_opt=None,
            lr=2e-4,
            recon_loss=Loss('mse', reduction="mean")
    ):
        super().__init__()
        self.btl_size = btl_size
        default_opt = torch.optim.Adam

        if encoder_and_opt is None:
            layer_list = [
                nn.Conv2d(input_channel, 16, kernel_size=1,stride=2), #conv1
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3,stride=2, padding=1), #conv3
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, btl_size, kernel_size=3,stride=2, padding=1), #conv8
                nn.BatchNorm2d(btl_size),
                nn.ReLU(),
            ]
            default_encoder = nn.Sequential(*layer_list)
            self.encoder = default_encoder
            self.encoder_opt = default_opt(self.encoder.parameters(), lr=lr)
        else:
            self.encoder = encoder_and_opt[0]
            self.encoder_opt = encoder_and_opt[1]
        
        if decoder_and_opt is None:
            layer_list = [
                nn.ConvTranspose2d(btl_size, 64, kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3,stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=3,stride=2, padding=1, output_padding=1),
            ]
            default_decoder = nn.Sequential(*layer_list)
            self.decoder = default_decoder
            self.decoder_opt = default_opt(self.decoder.parameters(), lr=lr)
        else:
            self.decoder = decoder_and_opt[0]
            self.decoder_opt = decoder_and_opt[1]
        
        self.recon_loss = recon_loss

    def encode(self, x):
        # |x| = (batch_size, #c, w, h)
        z = self.encoder(x)
        return z

    def decode(self, z):
        # |z| = (batch_size, btl_size)
        y = self.decoder(z)
        return y

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def get_loss_value(self, x, y, *args, **kwargs):
        output = self(x)
        loss = self.recon_loss(output, x)
        return loss

    @staticmethod
    def step(engine, mini_batch):
        # set model to train
        engine.model.train()
        engine.optimizer.zero_grad()

        # get data from mini_batch
        x, _ = mini_batch
        if engine.config.gpu_id >= 0:
            x = x.cuda(engine.config.gpu_id)
        # x = x.view(x.size(0), -1)

        # get loss value from model
        loss = engine.model.get_loss_value(x, x)

        # update parameters with loss
        loss.backward()

        engine.optimizer.step()

        return (float(loss), )

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, _ = mini_batch
            if engine.config.gpu_id >= 0:
                x = x.cuda(engine.config.gpu_id)
            # x = x.view(x.size(0), -1)

            loss = engine.model.get_loss_value(x, x)

        return (float(loss), )

    @staticmethod
    def attach(trainer, evaluator, config):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'recon')

        if config.verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(trainer, ['recon'])

        if config.verbose >= 1:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_recon = engine.state.metrics['recon']
                print("Epoch {} - loss={:.4e}"
                      .format(engine.state.epoch, avg_recon))

        RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'recon')

        if config.verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(evaluator, ['recon'])

        if config.verbose >= 1:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_recon = engine.state.metrics['recon']
                print("Validation - recon={:.4e} lowest_recon={:.4e}"
                      .format(avg_recon, engine.lowest_loss))