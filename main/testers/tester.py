import os
import torch
from torch.utils import tensorboard
import time, datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import ceil
from modules.utils import AverageMeter, matplotlib_imshow


class Tester:
    def __init__(self, args, dataloader, model, optimizer, loss_funcs: dict, metric_funcs: dict, device):
        self.args = args
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss_funcs = loss_funcs
        self.metric_funcs = metric_funcs
        self.device = device
        self._set_testing_constants()
        self._set_testing_variables()

    def test(self):
        self._restore_checkpoint()
        self.model.eval()
        self.end_time = time.time()
        print(len(self.dataloader.test_data_loader))
        for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.test_data_loader), total=len(self.dataloader.test_data_loader), desc='Test'):
            self.batch_idx = batch_idx
            self._test_step(batch_data, batch_label)
            if (batch_idx % self.TEST_LOG_INTERVAL == 0) or (batch_idx + 1 == len(self.dataloader.test_data_loader)):
                print('Test Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                self._log_progress()
        self.tensorboard_writer_test.close()

    def _test_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        with torch.no_grad():
            output_data = self.model(batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.test_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        total_loss_per_batch = 0
        for idx, loss_per_batch in enumerate(self.losses_per_batch.values()):
            total_loss_per_batch += loss_per_batch.mean()
        self.test_losses_per_epoch['total_loss'].update(total_loss_per_batch.item())
        # batch_diff_per_batch = batch_data - output_data
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
        if self.batch_idx == (len(self.dataloader.test_data_loader)-1):
            if "ROC" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['ROC'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data))
                self.metrics_per_batch['ROC'] = metric_value
                self.test_metrics_per_epoch['ROC'].update(metric_value)
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch)

    def _set_testing_constants(self):
        self.CHECKPOINT_SAVE_DIR = os.path.join(os.path.join(self.args.checkpoint_dir, self.args.model_name), self.args.exp_name)
        self.TENSORBOARD_LOG_SAVE_DIR = os.path.join(os.path.join(self.args.tensorboard_dir, self.args.model_name), self.args.exp_name)
        self.TIMEZONE = datetime.timezone(datetime.timedelta(hours=9))
        self.TEST_LOG_INTERVAL = ceil(len(self.dataloader.test_data_loader) / 10)
        if self.args.channel_num == 1:
            self.one_channel = True
        else:
            self.one_channel = False

    def _set_testing_variables(self):
        self.losses_per_batch = {}
        self.metrics_per_batch = {}
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.test_losses_per_epoch = {
            'total_loss': AverageMeter()
            }
        self.test_metrics_per_epoch = {}
        for loss_name in self.loss_funcs.keys():
            self.test_losses_per_epoch[loss_name] = AverageMeter()
        for metric_name in self.metric_funcs.keys():
            self.test_metrics_per_epoch[metric_name] = AverageMeter()
        self.diffs_per_data = []
        self.labels_per_data = []
        self.tensorboard_writer_test = tensorboard.SummaryWriter(os.path.join(self.TENSORBOARD_LOG_SAVE_DIR, 'test'), max_queue=100)

    def _restore_checkpoint(self):
        ckpts_list = os.listdir(self.CHECKPOINT_SAVE_DIR)
        if ckpts_list:
            last_epoch = sorted(list(map(int, [epoch.split('_')[-1].split('.')[0] for epoch in ckpts_list])))[-1]
            print('Restore Checkpoint epoch ', last_epoch)
            states = torch.load(os.path.join(self.CHECKPOINT_SAVE_DIR, 'epoch_{}.tar'.format(last_epoch)))
            self.epoch_idx = states['epoch']
            self.model.load_state_dict(states['model_state_dict'])
            self.optimizer.load_state_dict(states['optimizer_state_dict'])
        else:
            print('No checkpoints to restore')
    
    def _log_tensorboard(self, batch_data, batch_label, output_data, losses_per_batch, metrics_per_batch):
        losses_per_epoch = self.test_losses_per_epoch
        metric_per_epoch = self.test_metrics_per_epoch
        fig = plt.figure(figsize=(8, 8))
        for idx in range(self.args.test_tensorboard_shown_image_num):
            losses = []
            metrics = []
            random_sample_idx = np.random.randint(0, output_data.shape[0])
            ax_output = fig.add_subplot(2, self.args.test_tensorboard_shown_image_num, idx+self.args.test_tensorboard_shown_image_num+1, xticks=[], yticks=[])
            matplotlib_imshow(output_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            for loss_per_batch_name, loss_per_batch_value in losses_per_batch.items():
                losses.append(':'.join((loss_per_batch_name, str(round(loss_per_batch_value[random_sample_idx].mean().item(), 10)))))
            for metric_per_batch_name, metric_per_batch_value in metrics_per_batch.items():
                if metric_per_batch_name == 'ROC':
                    pass
                else:
                    metrics.append(':'.join((metric_per_batch_name, str(round(metric_per_batch_value[random_sample_idx].mean().item(), 10)))))
            if 'ROC' in metric_per_epoch.keys():
                metrics.append(':'.join((metric_per_batch_name, str(round(metric_per_epoch['ROC'].avg, 10)))))
            ax_output.set_title("Output\n" + "losses\n" + "\n".join(losses) + "\n\nmetrics\n"+ "\n".join(metrics) + "\nlabel: " + str(batch_label[random_sample_idx].item()))
            ax_batch = fig.add_subplot(2, self.args.test_tensorboard_shown_image_num, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(batch_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_batch.set_title("Ground Truth")
        plt.tight_layout()
        self.tensorboard_writer_test.add_figure("test", fig, global_step=self.epoch_idx)
        for loss_name, loss_value_per_epoch in losses_per_epoch.items():
            scalar_tag = [loss_name, '/loss']
            # print('tester loss scalar_tag: ', scalar_tag)
            self.tensorboard_writer_test.add_scalar(''.join(scalar_tag), loss_value_per_epoch.avg, self.epoch_idx)
            # print('tester loss loss_value_per_epoch.avg: ', loss_value_per_epoch.avg)
            # print('tester loss self.epoch_idx: ', self.epoch_idx)
        for metric_name, metric_value_per_epoch in metric_per_epoch.items():
            scalar_tag = [metric_name, '/metric']
            # print('tester metric scalar_tag: ', scalar_tag)
            self.tensorboard_writer_test.add_scalar(''.join(scalar_tag), metric_value_per_epoch.avg, self.epoch_idx)
            # print('tester metric metric_value_per_epoch.avg: ', metric_value_per_epoch.avg)
            # print('tester metric self.epoch_idx: ', self.epoch_idx)
        self.tensorboard_writer_test.flush()

    def _log_progress(self):
        self.data_time.update(time.time() - self.end_time)
        curr_time = datetime.datetime.now(self.TIMEZONE)
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        print("current time :\t",curr_time)
        print("epoch idx :\t", self.epoch_idx)
        print("batch index :\t", self.batch_idx)
        print("learning rate :\t", self.optimizer.param_groups[0]['lr'])
        print("batch_time :\t", self.batch_time.avg)
        print("data_time :\t", self.data_time.avg)
        print("losses")
        for loss_name, loss_value_per_epoch in self.test_losses_per_epoch.items():
            print(loss_name, ": ", loss_value_per_epoch.avg)
        print("metrics")
        for metric_name, metric_value_per_epoch in self.test_metrics_per_epoch.items():
            print(metric_name, ": ", metric_value_per_epoch.avg)