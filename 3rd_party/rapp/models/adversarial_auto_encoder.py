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

from models import AbstractModel
from modules import FCModule, Loss
from utils.common_utils import get_hidden_layer_sizes
import utils.common_utils as utils


class AdversarialAutoEncoder(AbstractModel):
    def __init__(
        self,
        input_size,
        n_layers=10,
        btl_size=100,
        encoder_and_opt=None,
        decoder_and_opt=None,
        discriminator_and_opt=None,
        lr=0.001,
        recon_loss=Loss('mse', reduction="sum"),
        bce_loss=Loss('bce'),
    ):
        super().__init__()
        self.btl_size = btl_size
        default_opt = torch.optim.Adam

        if encoder_and_opt is None:
            default_encoder = FCModule(
                input_size=input_size,
                output_size=btl_size,
                hidden_sizes=get_hidden_layer_sizes(input_size, btl_size, n_hidden_layers=n_layers - 1),
                use_batch_norm=True,
                act="leakyrelu",
                last_act=None,
            )
            self.encoder = default_encoder
            self.encoder_opt = default_opt(self.encoder.parameters(), lr=lr)
        else:
            self.encoder = encoder_and_opt[0]
            self.encoder_opt = encoder_and_opt[1]

        if decoder_and_opt is None:
            default_decoder = FCModule(
                input_size=btl_size,
                output_size=input_size,
                hidden_sizes=get_hidden_layer_sizes(btl_size, input_size, n_hidden_layers=n_layers - 1),
                use_batch_norm=True,
                act="leakyrelu",
                last_act=None,
            )
            self.decoder = default_decoder
            self.decoder_opt = default_opt(self.decoder.parameters(), lr=lr)
        else:
            self.decoder = decoder_and_opt[0]
            self.decoder_opt = decoder_and_opt[1]

        if discriminator_and_opt is None:
            default_discriminator = FCModule(
                input_size=btl_size,
                output_size=1,
                hidden_sizes=get_hidden_layer_sizes(btl_size, 1, n_hidden_layers=n_layers - 1),
                use_batch_norm=False,
                dropout_p=0.2,
                act="relu",
                last_act="sigmoid"
            )
            self.discriminator = default_discriminator
            self.discriminator_opt = default_opt(self.discriminator.parameters(), lr=lr / 2)
        else:
            self.discriminator = discriminator_and_opt[0]
            self.discriminator_opt = discriminator_and_opt[1]

        self.recon_loss = recon_loss
        self.bce_loss = bce_loss

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_recon_loss(self, x, y, *args, **kwargs):
        x_recon = self.forward(x)
        recon_loss = self.recon_loss(x_recon, x)
        return recon_loss

    def get_D_loss(self, x, y, *args, **kwargs):
        device = x.device

        z_fake = self.encoder(x)
        z_true = torch.randn(x.size(0), self.btl_size).to(device)

        z_true_pred = self.discriminator(z_true)
        z_fake_pred = self.discriminator(z_fake)

        target_ones = torch.ones(x.size(0), 1).to(device)
        target_zeros = torch.zeros(x.size(0), 1).to(device) 

        true_loss = self.bce_loss(z_true_pred, target_ones)
        fake_loss = self.bce_loss(z_fake_pred, target_zeros)

        D_loss = true_loss + fake_loss
        return D_loss

    def get_G_loss(self, x, y, *args, **kwargs):
        target_ones = torch.ones(x.size(0), 1).to(x.device)
        z_fake = self.encoder(x)
        z_fake_pred = self.discriminator(z_fake)
        G_loss = self.bce_loss(z_fake_pred, target_ones)
        return G_loss

    @staticmethod
    def step(engine, mini_batch):
        engine.model.train()

        x, _ = mini_batch
        if engine.config.gpu_id >= 0:
            x = x.cuda(engine.config.gpu_id)

        # learning by back prop
        
        # update: decoder, encoder
        engine.model.decoder_opt.zero_grad()
        engine.model.encoder_opt.zero_grad()
        recon_loss = engine.model.get_recon_loss(x, None)
        recon_loss.backward()
        engine.model.decoder_opt.step()
        engine.model.encoder_opt.step()

        # update: discriminator
        engine.model.discriminator_opt.zero_grad()
        D_loss = engine.model.get_D_loss(x, None)
        D_loss.backward()
        engine.model.discriminator_opt.step()

        # update: generator
        engine.model.encoder_opt.zero_grad()
        G_loss = engine.model.get_G_loss(x, None)
        G_loss.backward()
        engine.model.encoder_opt.step()

        return float(recon_loss), float(D_loss), float(G_loss)

    @staticmethod
    def validate(engine, mini_batch):
        # idx_loss = 0
        engine.model.eval()

        with torch.no_grad():
            x, _ = mini_batch
            if engine.config.gpu_id >= 0:
                x = x.cuda(engine.config.gpu_id)
            recon_loss = engine.model.get_recon_loss(x, None)
            D_loss = engine.model.get_D_loss(x, None)
            G_loss = engine.model.get_G_loss(x, None)

        return float(recon_loss), float(D_loss), float(G_loss)

    @staticmethod
    def attach(trainer, evaluator, config):
        from ignite.engine import Events
        from ignite.metrics import RunningAverage
        from ignite.contrib.handlers.tqdm_logger import ProgressBar

        #
        # trainer
        #

        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'recon')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'D_loss')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'G_loss')

        if config.verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(trainer, ['recon', 'G_loss', 'D_loss'])

        if config.verbose >= 1:
            @trainer.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                recon_loss = engine.state.metrics['recon']
                D_loss = engine.state.metrics['D_loss']
                G_loss = engine.state.metrics['G_loss']
                print("Epoch {} - recon={:.4e} d_loss={:.4e} g_loss={:.4e}"
                      .format(engine.state.epoch, recon_loss, D_loss, G_loss))

        #
        # evaluator
        #
        RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'recon')
        RunningAverage(output_transform=lambda x: x[1]).attach(evaluator, 'D_loss')
        RunningAverage(output_transform=lambda x: x[2]).attach(evaluator, 'G_loss')

        if config.verbose >= 2:
            pbar = ProgressBar()
            pbar.attach(evaluator, ['recon'])

        if config.verbose >= 1:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                recon_loss = engine.state.metrics['recon']
                D_loss = engine.state.metrics['D_loss']
                G_loss = engine.state.metrics['G_loss']
                print("Validation - recon={:.4e} d_loss={:.4e} g_loss={:.4e} lowest_recon={:.4e}"
                      .format(recon_loss, D_loss, G_loss, engine.lowest_loss))
