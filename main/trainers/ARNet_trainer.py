from modules.utils import AverageMeter, matplotlib_imshow
from main.trainers.trainer import Trainer
from numpy.random import randint
import time
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.titlesize': 'small'})


class ARNetTrainer(Trainer):
    def __init__(self, args, dataloader, model, optimizer, lr_scheduler, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, lr_scheduler, loss_funcs, metric_funcs, device)
        self.angles = [0, 90, 180, 270]
        self.rotate_funtions = {
            '0': lambda x, d1, d2: x,
            '90': lambda x, d1, d2: x.transpose(d1, d2).flip(d1),
            '180': lambda x, d1, d2: x.flip(d1).flip(d2),
            '270': lambda x, d1, d2: x.transpose(d1, d2).flip(d2)
            }

    def _train_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)    
        self.optimizer.zero_grad()
        # Random rotation: This operation rotates x anticlockwise by angle alpha around the center of each image channel. The rotation angle alpha is randomly selected from a set {0, 90, 180, 270}
        angle = self.angles[randint(0, 4)]
        transformed_batch_data = self.rotate_funtions[str(angle)](batch_data, -2, -1)
        if not transformed_batch_data.is_contiguous():
            transformed_batch_data = transformed_batch_data.contiguous()
        # Graying: This operation averages each pixel value along the channel dimension of images.
        if transformed_batch_data.shape[1] == 3:
            transformed_batch_data = torch.mean(transformed_batch_data, dim=1, keepdim=True) # channel dim 1
        batch_data = batch_data.to(self.device)
        transformed_batch_data = transformed_batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        output_data = self.model(transformed_batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.train_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        args_for_losses = {
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
            self._log_tensorboard(batch_data, batch_label, transformed_batch_data, output_data, self.losses_per_batch, self.metrics_per_batch)

    def _val_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
        # Random rotation: This operation rotates x anticlockwise by angle alpha around the center of each image channel. The rotation angle alpha is randomly selected from a set {0, 90, 180, 270}
        angle = self.angles[randint(0, 4)]
        transformed_batch_data = self.rotate_funtions[str(angle)](batch_data, -2, -1)
        if not transformed_batch_data.is_contiguous():
            transformed_batch_data = transformed_batch_data.contiguous()
        # Graying: This operation averages each pixel value along the channel dimension of images.
        if transformed_batch_data.shape[1] == 3:
            transformed_batch_data = torch.mean(transformed_batch_data, dim=1, keepdim=True) # channel dim 1
        batch_data = batch_data.to(self.device)
        transformed_batch_data = transformed_batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        with torch.no_grad():
            output_data = self.model(transformed_batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.valid_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        args_for_losses = {
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
            self._log_tensorboard(batch_data, batch_label, transformed_batch_data, output_data, self.losses_per_batch, self.metrics_per_batch, True)
    
    def _set_training_variables(self):
        super()._set_training_variables()
        for loss_name in self.model.get_losses_name():
            self.train_losses_per_epoch[loss_name] = AverageMeter()
            self.valid_losses_per_epoch[loss_name] = AverageMeter()
    
    def _log_tensorboard(self, batch_data, batch_label, transformed_batch_data, output_data, losses_per_batch, metrics_per_batch, is_valid=False):
        training_state = "train"
        losses_per_epoch = self.train_losses_per_epoch
        metric_per_epoch = self.train_metrics_per_epoch
        if is_valid:
            training_state = "valid"
            losses_per_epoch = self.valid_losses_per_epoch
            metric_per_epoch = self.valid_metrics_per_epoch
        fig = plt.figure(figsize=(8, 12))
        for idx in range(self.args.train_tensorboard_shown_image_num):
            losses = []
            metrics = []
            random_sample_idx = randint(0, output_data.shape[0])
            ax_output = fig.add_subplot(3, self.args.train_tensorboard_shown_image_num, idx+self.args.train_tensorboard_shown_image_num+1, xticks=[], yticks=[])
            matplotlib_imshow(output_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            for loss_per_batch_name, loss_per_batch_value in losses_per_batch.items():
                losses.append(':'.join((loss_per_batch_name, str(round(loss_per_batch_value[random_sample_idx].mean().item(), 10)))))
            for metric_per_batch_name, metric_per_batch_value in metrics_per_batch.items():
                metrics.append(':'.join((metric_per_batch_name, str(round(metric_per_batch_value[random_sample_idx].mean().item(), 10)))))
            ax_output.set_title("Output\n" + "losses\n" + "\n".join(losses) + "\n\nmetrics\n"+ "\n".join(metrics) + "\nlabel: " + str(batch_label[random_sample_idx].item()))
            ax_batch = fig.add_subplot(3, self.args.train_tensorboard_shown_image_num, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(batch_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_batch.set_title("Ground Truth")
            ax_transfomred_batch = fig.add_subplot(3, self.args.train_tensorboard_shown_image_num, idx+self.args.train_tensorboard_shown_image_num*2+1, xticks=[], yticks=[])
            matplotlib_imshow(transformed_batch_data[random_sample_idx], one_channel=True, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_transfomred_batch.set_title("Transformed Batch Data")
        plt.tight_layout()
        if is_valid:
            self.tensorboard_writer_valid.add_figure(training_state, fig, global_step=self.epoch_idx)
        else:
            self.tensorboard_writer_train.add_figure(training_state, fig, global_step=self.epoch_idx)
        # log losses
        for loss_name, loss_value_per_epoch in losses_per_epoch.items():
            scalar_tag = [loss_name, '/loss']
            if is_valid:
                self.tensorboard_writer_valid.add_scalar(''.join(scalar_tag), loss_value_per_epoch.avg, self.epoch_idx)
            else:
                self.tensorboard_writer_train.add_scalar(''.join(scalar_tag), loss_value_per_epoch.avg, self.epoch_idx)
        # log metrics
        for metric_name, metric_value_per_epoch in metric_per_epoch.items():
            scalar_tag = [metric_name, '/metric']
            if is_valid:
                self.tensorboard_writer_valid.add_scalar(''.join(scalar_tag), metric_value_per_epoch.avg, self.epoch_idx)
            else:
                self.tensorboard_writer_train.add_scalar(''.join(scalar_tag), metric_value_per_epoch.avg, self.epoch_idx)
        self.tensorboard_writer_train.flush()
        self.tensorboard_writer_valid.flush()