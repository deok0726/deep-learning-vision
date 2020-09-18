from main.trainer.trainer import Trainer
from models.AE import autoencoder
class AE_trainer(Trainer):
    def __init__(self, args, model: autoencoder):
        self.model = autoencoder
        # self.params = load_dh_params(dhfile)

    def forward(self, input: Tensor):
        return pass
        
    def train(self):
        super().train()
    def _training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        return pass

    def save_images(self):
        return pass

    def configure_optimizers(self):
        return pass

    def _set_training_variables(self, args):
        super()._set_training_variables(args)

    # # data_loader
    # def train_dataloader(self):
    #     return pass
    # def val_dataloader(self):
    #     return pass
    # def data_transforms(self):
    #     return pass