from tqdm import tqdm
from modules.utils import AverageMeter, matplotlib_imshow
from math import ceil
import time, datetime
import torch
from torch.utils import tensorboard
import torchvision
import os
import matplotlib.pyplot as plt
import copy
plt.rcParams.update({'axes.titlesize': 'small'})
from sklearn import metrics
# import numpy as np


class Trainer:
    def __init__(self, args, dataloader, model, optimizer, lr_scheduler, loss_funcs: dict, metric_funcs: dict, device):
        self.args = args
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_funcs = loss_funcs
        self.metric_funcs = metric_funcs
        self.device = device
        self._set_training_constants()
        self._set_training_variables()

    def train(self):
        self.epoch_idx = 0
        self._restore_checkpoint()
        self.global_step = self.epoch_idx*len(self.dataloader.train_data_loader)
        self.tensorboard_writer_train.add_graph(self.model, self.dataloader.sample_train_data.to(self.device))
        self.tensorboard_writer_valid.add_graph(self.model, self.dataloader.sample_train_data.to(self.device))
        for epoch_idx in tqdm(range(self.args.num_epoch), desc='Train'):
            # ================================================================== #
            #                         training                                   #
            # ================================================================== #
            self.model.train()
            self.end_time = time.time()
            # print(len(self.dataloader.train_data_loader))
            for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.train_data_loader), total=len(self.dataloader.train_data_loader), desc='Epoch %d' % self.epoch_idx):
                self.batch_idx = batch_idx
                self.global_step += 1
                self._train_step(batch_data, batch_label)
                if batch_idx % self.TRAIN_LOG_INTERVAL == 0:
                    print('Train Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                    self._log_progress()
                # if (batch_idx + 1 == len(self.dataloader.train_data_loader)) and (epoch_idx + 1 == self.args.num_epoch):
                #     self.last_train_losses = copy.deepcopy(self.train_losses_per_epoch)
                #     self.last_train_metrics = copy.deepcopy(self.train_metrics_per_epoch)
            self._reset_training_variables()
            # ================================================================== #
            #                         validating                                 #
            # ================================================================== #
            self.model.eval()
            self.end_time = time.time()
            for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.valid_data_loader), total=len(self.dataloader.valid_data_loader), desc='Epoch %d' % self.epoch_idx):
                self.batch_idx = batch_idx
                self._val_step(batch_data, batch_label)
                if batch_idx % self.VALID_LOG_INTERVAL == 0:
                    print('Valid Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                    self._log_progress(is_valid=True)
                if (batch_idx + 1 == len(self.dataloader.valid_data_loader)) and (epoch_idx + 1 == self.args.num_epoch):
                    self.last_valid_losses = copy.deepcopy(self.valid_losses_per_epoch)
                    self.last_valid_metrics = copy.deepcopy(self.valid_metrics_per_epoch)
            self._reset_training_variables(is_valid=True)
            self._save_checkpoint()
            self.epoch_idx += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        self._log_hparams()

    def _train_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
    
        self.optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        output_data = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.train_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
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
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.valid_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        self.valid_losses_per_epoch['total_loss'].update(valid_losses_per_epoch.item())
        for metric_func_name, metric_func in self.metric_funcs.items():
            metric_value = metric_func(batch_data, output_data)
            self.metrics_per_batch[metric_func_name] = metric_value
            self.valid_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.valid_data_loader)-1:
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch, True)
    
    def _set_training_constants(self):
        self.CHECKPOINT_SAVE_DIR = os.path.join(os.path.join(self.args.checkpoint_dir, self.args.model_name), self.args.exp_name)
        self.TENSORBOARD_LOG_SAVE_DIR = os.path.join(os.path.join(self.args.tensorboard_dir, self.args.model_name), self.args.exp_name)
        self.TIMEZONE = datetime.timezone(datetime.timedelta(hours=9))
        self.TRAIN_LOG_INTERVAL = ceil(len(self.dataloader.train_data_loader) / 10)
        self.VALID_LOG_INTERVAL = ceil(len(self.dataloader.valid_data_loader) / 10)
        if self.args.channel_num == 1:
            self.one_channel = True
        else:
            self.one_channel = False

    def _set_training_variables(self):
        self.losses_per_batch = {}
        self.metrics_per_batch = {}
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses_per_epoch = {
            'total_loss': AverageMeter()
        }
        self.train_metrics_per_epoch = {}
        self.valid_losses_per_epoch = {
            'total_loss': AverageMeter()
        }
        self.valid_metrics_per_epoch = {}
        for loss_name in self.loss_funcs.keys():
            self.train_losses_per_epoch[loss_name] = AverageMeter()
            self.valid_losses_per_epoch[loss_name] = AverageMeter()
        for metric_name in self.metric_funcs.keys():
            self.train_metrics_per_epoch[metric_name] = AverageMeter()
            self.valid_metrics_per_epoch[metric_name] = AverageMeter()
        self.tensorboard_writer_train = tensorboard.SummaryWriter(os.path.join(self.TENSORBOARD_LOG_SAVE_DIR, 'train'), max_queue=100)
        self.tensorboard_writer_valid = tensorboard.SummaryWriter(os.path.join(self.TENSORBOARD_LOG_SAVE_DIR, 'valid'), max_queue=100)
    
    def _reset_training_variables(self, is_valid=False):
        self.batch_time.reset()
        self.data_time.reset()
        if is_valid:
            for loss_name in self.valid_losses_per_epoch.keys():
                self.valid_losses_per_epoch[loss_name].reset()
            for metric_name in self.metric_funcs.keys():
                self.valid_metrics_per_epoch[metric_name].reset()
        else:
            for loss_name in self.train_losses_per_epoch.keys():
                self.train_losses_per_epoch[loss_name].reset()
            for metric_name in self.metric_funcs.keys():
                self.train_metrics_per_epoch[metric_name].reset()

    def _save_checkpoint(self):
        states = {
            'epoch': self.epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }
        torch.save(states, os.path.join(self.CHECKPOINT_SAVE_DIR, 'epoch_{}.tar'.format(self.epoch_idx)))
        ckpts_list = os.listdir(self.CHECKPOINT_SAVE_DIR)
        if ckpts_list and (len(ckpts_list) > 1): # save latest checkpoint only
            oldest_epoch = sorted(list(map(int, [epoch.split('_')[-1].split('.')[0] for epoch in ckpts_list])))[0]
            os.remove(os.path.join(self.CHECKPOINT_SAVE_DIR, 'epoch_{}.tar'.format(oldest_epoch)))
        print('Checkpoint saved epoch ', self.epoch_idx)

    def _restore_checkpoint(self):
        ckpts_list = os.listdir(self.CHECKPOINT_SAVE_DIR)
        if ckpts_list:
            last_epoch = sorted(list(map(int, [epoch.split('_')[-1].split('.')[0] for epoch in ckpts_list])))[-1]
            print('Restore Checkpoint epoch ', last_epoch)
            states = torch.load(os.path.join(self.CHECKPOINT_SAVE_DIR, 'epoch_{}.tar'.format(last_epoch)))
            self.epoch_idx = states['epoch'] + 1
            self.model.load_state_dict(states['model_state_dict'])
            self.optimizer.load_state_dict(states['optimizer_state_dict'])
        else:
            print('No checkpoints to restore')

    def _log_tensorboard(self, batch_data, batch_label, output_data, losses_per_batch, metrics_per_batch, is_valid=False):
        training_state = "train"
        losses_per_epoch = self.train_losses_per_epoch
        metric_per_epoch = self.train_metrics_per_epoch
        if is_valid:
            training_state = "valid"
            losses_per_epoch = self.valid_losses_per_epoch
            metric_per_epoch = self.valid_metrics_per_epoch
        fig = plt.figure(figsize=(8, 8))
        for idx in range(self.args.train_tensorboard_shown_image_num):
            losses = []
            metrics = []
            ax_output = fig.add_subplot(2, self.args.train_tensorboard_shown_image_num, idx+self.args.train_tensorboard_shown_image_num+1, xticks=[], yticks=[])
            matplotlib_imshow(output_data[idx], one_channel=self.one_channel)
            for loss_per_batch_name, loss_per_batch_value in losses_per_batch.items():
                losses.append(':'.join((loss_per_batch_name, str(round(loss_per_batch_value[idx].mean().item(), 10)))))
            for metric_per_batch_name, metric_per_batch_value in metrics_per_batch.items():
                metrics.append(':'.join((metric_per_batch_name, str(round(metric_per_batch_value[idx].mean().item(), 10)))))
            # ax_output.set_title("Output\n" + "\n".join(losses) + "\nlabel: " + str(batch_label[idx].item()))
            ax_output.set_title("Output\n" + "losses\n" + "\n".join(losses) + "\n\nmetrics\n"+ "\n".join(metrics) + "\nlabel: " + str(batch_label[idx].item()))
            ax_batch = fig.add_subplot(2, self.args.train_tensorboard_shown_image_num, idx+1, xticks=[], yticks=[])
            # if training_state == "train":
                # torchvision.utils.save_image(batch_data[idx].double(), "/root/anomaly_detection/temp/batch_" + str(idx) + "_" + str(idx) + "_" + str(batch_label[idx]) + ".png", "PNG")
                # torchvision.utils.save_image(output_data[idx].double(), "/root/anomaly_detection/temp/output_" + str(idx) + "_" + str(idx) + "_" + str(batch_label[idx]) + ".png", "PNG")
            matplotlib_imshow(batch_data[idx], one_channel=self.one_channel)
            ax_batch.set_title("Ground Truth")
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
        
    def _log_hparams(self):
        # train_losses_and_metrics = {}
        valid_losses_and_metrics = {}
        # for loss_name, loss_value_last_epoch in self.last_train_losses.items():
        #     train_losses_and_metrics[loss_name + '/loss'] = loss_value_last_epoch.avg
        # for metric_name, metric_value_last_epoch in self.last_train_metrics.items():
        #     train_losses_and_metrics[metric_name + '/metric'] = metric_value_last_epoch.avg
        for loss_name, loss_value_last_epoch in self.last_valid_losses.items():
            valid_losses_and_metrics[loss_name + '/loss'] = loss_value_last_epoch.avg
        for metric_name, metric_value_last_epoch in self.last_valid_metrics.items():
            valid_losses_and_metrics[metric_name + '/metric'] = metric_value_last_epoch.avg
        args_dict = vars(self.args)
        args_dict['trained_epoch'] = self.epoch_idx
        self.tensorboard_writer_valid.add_hparams(args_dict, valid_losses_and_metrics)
        # self.tensorboard_writer_train.add_hparams(args_dict, train_losses_and_metrics)

    def _log_progress(self, is_valid=False):
        self.data_time.update(time.time() - self.end_time)
        curr_time = datetime.datetime.now(self.TIMEZONE)
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        print("current time :\t",curr_time)
        print("epoch index :\t",self.epoch_idx)
        print("batch index :\t",self.batch_idx)
        print("learning rate :\t",self.optimizer.param_groups[0]['lr'])
        print("batch_time :\t",self.batch_time.avg)
        print("data_time :\t",self.data_time.avg)
        print("losses")
        if is_valid:
            for loss_name, loss_value_per_epoch in self.valid_losses_per_epoch.items():
                print(loss_name, ": ", loss_value_per_epoch.avg)
        else:
            for loss_name, loss_value_per_epoch in self.train_losses_per_epoch.items():
                print(loss_name, ": ", loss_value_per_epoch.avg)
        print("metrics")
        if is_valid:
            for metric_name, metric_value_per_epoch in self.valid_metrics_per_epoch.items():
                print(metric_name, ": ", metric_value_per_epoch.avg)
        else:
            for metric_name, metric_value_per_epoch in self.train_metrics_per_epoch.items():
                print(metric_name, ": ", metric_value_per_epoch.avg)
