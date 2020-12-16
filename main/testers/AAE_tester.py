from modules.utils import AverageMeter
from main.testers.tester import Tester as main_tester
import time
import torch
import numpy as np

class Tester(main_tester):
    def __init__(self, args, dataloader, model, optimizer, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, loss_funcs, metric_funcs, device)
    
    def _test_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)

        # self.optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        # output_data = self.model(batch_data)
        output_data, discriminator_z_fake, discriminator_z_real  = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            if loss_func_name == 'discriminator':
                loss_values = loss_func(discriminator_z_fake, discriminator_z_real)
            else:
                loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.test_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        #
        total_kld, kld_loss= self.model.set_kld_loss(self.losses_per_batch['MSE'])
        self.losses_per_batch['total_kld'] = total_kld
        self.test_losses_per_epoch['total_kld'].update(total_kld.mean().item())
        
        ###
        total_loss_per_batch = 0
        for idx, loss_per_batch in enumerate(self.losses_per_batch.values()):
            total_loss_per_batch += loss_per_batch.mean()
        self.test_losses_per_epoch['total_loss'].update(total_loss_per_batch.item())
        batch_diff_per_batch = (batch_data - output_data) ** 2
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        self.diffs_per_data.extend(batch_diff_per_batch.cpu().detach().numpy())
        self.labels_per_data.extend(batch_label.cpu().detach().numpy())
        for metric_func_name, metric_func in self.metric_funcs.items():
            if metric_func_name == 'ROC':
                pass
            else:
                metric_value = metric_func(batch_data, output_data)
                self.metrics_per_batch[metric_func_name] = metric_value
                self.test_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.test_data_loader)-1:
            if "ROC" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['ROC'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data))
                self.metrics_per_batch['ROC'] = metric_value
                self.test_metrics_per_epoch['ROC'].update(metric_value)
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch, True)
