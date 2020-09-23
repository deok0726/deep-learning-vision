from tqdm import tqdm
from utils.utils import AverageMeter, automkdir
from math import ceil
import time, datetime
import torch
import os

class Trainer:
    def __init__(self, args, dataloader, model, loss_funcs: dict, optimizer, metric_funcs: dict, device):
        self.args = args
        self.dataloader = dataloader
        self.model = model
        self.loss_funcs = loss_funcs
        self.optimizer = optimizer
        self.metric_funcs = metric_funcs
        self.device = device
        self._set_training_constants()
        self._set_training_variables()

    def train(self):
        self.epoch_idx = 0
        self._restore_checkpoint()
        for epoch_idx in tqdm(range(self.args.num_epoch), desc='Train'):
            # ================================================================== #
            #                         training                                   #
            # ================================================================== #
            self.model.train()
            self.end_time = time.time()
            for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.train_data_loader), total=len(self.dataloader.train_data_loader), desc='Epoch %d' % self.epoch_idx):
                self.batch_idx = batch_idx
                self._train_step(batch_data, batch_label)                
                if batch_idx % self.TRAIN_LOG_INTERVAL == 0:
                    print('Train Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                    self._log_progress()
            self._reset_training_variables()
            # ================================================================== #
            #                         validating                                 #
            # ================================================================== #
            self.model.eval()
            self.end_time = time.time()
            for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.valid_data_loader), total=len(self.dataloader.valid_data_loader), desc='Epoch %d' % self.epoch_idx):
                self.batch_idx = batch_idx
                self._val_step(batch_data, batch_label)
                # log_interval = len(self.dataloader.valid_data_loader) // 10
                if batch_idx % self.VALID_LOG_INTERVAL == 0:
                    print('Valid Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                    self._log_progress(is_valid=True)
            self._reset_training_variables(is_valid=True)
            self._save_checkpoint()
            self.epoch_idx += 1

    def _train_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
    
        self.optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        output_data = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_value = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_value
            self.train_losses_per_epoch[loss_func_name].update(loss_value.item())
        for metric_func_name, metric_func in self.metric_funcs.items():
            metric_value = metric_func(batch_data, output_data)
            # self.metrics_per_batch[metric_func_name] = metric_value
            self.train_metrics_per_epoch[metric_func_name].update(metric_value.item())
        total_loss_per_batch = 0
        for loss_per_batch in self.losses_per_batch.values():
            total_loss_per_batch += loss_per_batch
        total_loss_per_batch.backward()
        self.optimizer.step()
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()

    def _val_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        with torch.no_grad():
            output_data = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_value = loss_func(batch_data, output_data)
            # self.losses_per_batch[loss_func_name] = loss_value
            self.valid_losses_per_epoch[loss_func_name].update(loss_value.item())
        for metric_func_name, metric_func in self.metric_funcs.items():
            metric_value = metric_func(batch_data, output_data)
            # self.metrics_per_batch[metric_func_name] = metric_value
            self.valid_metrics_per_epoch[metric_func_name].update(metric_value.item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
    
    def _set_training_constants(self):
        self.CHECKPOINT_SAVE_DIR = os.path.join(self.args.save_dir, self.args.model_name)
        automkdir(self.CHECKPOINT_SAVE_DIR)
        self.TIMEZONE = datetime.timezone(datetime.timedelta(hours=9))
        self.TRAIN_LOG_INTERVAL = ceil(len(self.dataloader.train_data_loader) / 10)
        self.VALID_LOG_INTERVAL = ceil(len(self.dataloader.valid_data_loader) / 10)

    def _set_training_variables(self):
        self.losses_per_batch = {}
        # self.metrics_per_batch = {}
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses_per_epoch = {}
        self.train_metrics_per_epoch = {}
        self.valid_losses_per_epoch = {}
        self.valid_metrics_per_epoch = {}
        for loss_name in self.loss_funcs.keys():
            self.train_losses_per_epoch[loss_name] = AverageMeter()
            self.valid_losses_per_epoch[loss_name] = AverageMeter()
        for metric_name in self.metric_funcs.keys():
            self.train_metrics_per_epoch[metric_name] = AverageMeter()
            self.valid_metrics_per_epoch[metric_name] = AverageMeter()
    
    def _reset_training_variables(self, is_valid=False):
        self.batch_time.reset()
        self.data_time.reset()
        if is_valid:
            for loss_name in self.loss_funcs.keys():
                self.valid_losses_per_epoch[loss_name].reset()
            for metric_name in self.metric_funcs.keys():
                self.valid_metrics_per_epoch[metric_name].reset()
        else:
            for loss_name in self.loss_funcs.keys():
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

    def _log_tensorboard(self):
        raise NotImplementedError

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
