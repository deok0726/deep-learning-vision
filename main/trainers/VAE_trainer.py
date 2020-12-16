from main.trainers.trainer import Trainer

import time
import torch

# from models.AE import autoencoder
class VAE_trainer(Trainer):
    def __init__(self, args, dataloader, model, optimizer, lr_scheduler, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, lr_scheduler, loss_funcs, metric_funcs, device)
        # self.model = autoencoder
        # self.params = load_dh_params(dhfile)

    # def forward(self, input: Tensor):
    #     super().
    #     return pass
        
    # def train(self):
    #     super().train()
    def _train_step(self, batch_data, batch_label, optimizer_idx = 0):
        self.data_time.update(time.time() - self.end_time)

        self.optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        output_data = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.train_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        
        # kld_custom, total loss 
        ## [Todo] self.loss_funcs에 넣기, 
        ## 지금은 total loss안에도 recon loss가 있어서 한 배치에서 recon loss 두번 update함
        # total_kld, kld_loss= self.model.set_kld_loss(output_data, batch_data)
        total_kld, kld_loss= self.model.set_kld_loss(self.losses_per_batch['MSE'])
        self.losses_per_batch['total_kld'] = total_kld
        self.train_losses_per_epoch['total_kld'].update(total_kld.mean().item())
        # self.losses_per_batch['kld_loss'] = kld_loss
        # self.train_losses_per_epoch['kld_loss'].update(kld_loss.mean().item())
        
        # discriminator loss

        total_loss_per_batch = 0
        batch_diff_per_batch = batch_data - output_data
        for idx, loss_per_batch in enumerate(self.losses_per_batch.values()):
            total_loss_per_batch += loss_per_batch.mean()
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        total_loss_per_batch.backward()
        for metric_func_name, metric_func in self.metric_funcs.items():
            if metric_func_name == 'ROC':
                try:
                    metric_value = metric_func(batch_diff_per_batch.cpu().detach().numpy(), batch_label.cpu().detach().numpy())
                except Exception as e:
                    metric_value = 0
                    print(e)
            else:
                metric_value = metric_func(batch_data, output_data)
            self.metrics_per_batch[metric_func_name] = metric_value
            self.train_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.optimizer.step()
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.train_data_loader)-1:
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch)

    def _val_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        with torch.no_grad():
            output_data = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.valid_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        batch_diff_per_batch = batch_data - output_data
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        for metric_func_name, metric_func in self.metric_funcs.items():
            if metric_func_name == 'ROC':
                try:
                    metric_value = metric_func(batch_diff_per_batch.cpu().detach().numpy(), batch_label.cpu().detach().numpy())
                except Exception as e:
                    print(e)
            else:
                metric_value = metric_func(batch_data, output_data)
            self.metrics_per_batch[metric_func_name] = metric_value
            self.valid_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.valid_data_loader)-1:
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch, True)
        
        

    # def save_images(self):
    #     return pass

    # def configure_optimizers(self):
    #     return pass

    # def _set_training_variables(self, args):
    #     super()._set_training_variables(args)