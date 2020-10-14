from modules.utils import AverageMeter
from main.trainers.trainer import Trainer
import time
import torch

class MemAETrainer(Trainer):
    def __init__(self, args, dataloader, model, optimizer, lr_scheduler, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, lr_scheduler, loss_funcs, metric_funcs, device)

    def _train_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
    
        self.optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        output_data = self.model(batch_data)
        mem_weight = getattr(output_data, 'mem_weight')
        output_data = getattr(output_data ,'output')
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.train_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        args_for_losses = {
            'batch_size':self.args.train_batch_size,
            'weight': mem_weight,
            'entropy_loss_coef': self.args.entropy_loss_coef,
            'x': batch_data,
            'y': output_data
            }
        losses = self.model.get_losses(args_for_losses)
        for loss_name, loss_values in losses.items():
            self.losses_per_batch[loss_name] = loss_values
            self.train_losses_per_epoch[loss_name].update(loss_values.mean().item())
        total_loss_per_batch = 0
        for idx, loss_per_batch in enumerate(self.losses_per_batch.values()):
            total_loss_per_batch += loss_per_batch.mean()
        self.train_losses_per_epoch['total_loss'].update(total_loss_per_batch.item())
        total_loss_per_batch.backward()
        for metric_func_name, metric_func in self.metric_funcs.items():
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
            mem_weight = getattr(output_data, 'mem_weight')
            output_data = getattr(output_data ,'output')
            # mem_weight = output_data['mem_weight']
            # output_data = output_data['output']
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.valid_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        args_for_losses = {
            'batch_size':self.args.train_batch_size,
            'weight': mem_weight,
            'entropy_loss_coef': self.args.entropy_loss_coef,
            'x': batch_data,
            'y': output_data
            }
        losses = self.model.get_losses(args_for_losses)
        for loss_name, loss_values in losses.items():
            self.losses_per_batch[loss_name] = loss_values
            self.valid_losses_per_epoch[loss_name].update(loss_values.mean().item())
        total_loss_per_batch = 0
        for idx, loss_per_batch in enumerate(self.losses_per_batch.values()):
            total_loss_per_batch += loss_per_batch.mean()
        self.valid_losses_per_epoch['total_loss'].update(total_loss_per_batch.item())
        for metric_func_name, metric_func in self.metric_funcs.items():
            metric_value = metric_func(batch_data, output_data)
            self.metrics_per_batch[metric_func_name] = metric_value
            self.valid_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.valid_data_loader)-1:
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch, True)
    
    def _set_training_variables(self):
        super()._set_training_variables()
        for loss_name in self.model.get_losses_name():
            self.train_losses_per_epoch[loss_name] = AverageMeter()
            self.valid_losses_per_epoch[loss_name] = AverageMeter()