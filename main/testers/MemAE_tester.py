from modules.utils import AverageMeter
from main.testers.tester import Tester
import time
import torch
import numpy as np

class MemAETester(Tester):
    def __init__(self, args, dataloader, model, optimizer, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, loss_funcs, metric_funcs, device)
    
    def _test_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        with torch.no_grad():
            output_data = self.model(batch_data)
            mem_weight = getattr(output_data, 'mem_weight')
            embedding = getattr(output_data, 'embedding')
            output_data = getattr(output_data ,'output')
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.test_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
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
            self.test_losses_per_epoch[loss_name].update(loss_values.mean().item())
        total_loss_per_batch = 0
        for idx, loss_per_batch in enumerate(self.losses_per_batch.values()):
            total_loss_per_batch += loss_per_batch.mean()
        self.test_losses_per_epoch['total_loss'].update(total_loss_per_batch.item())
        batch_diff_per_batch = (batch_data - output_data) ** 2
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        self.diffs_per_data.extend(batch_diff_per_batch.cpu().detach().numpy())
        self.labels_per_data.extend(batch_label.cpu().detach().numpy())
        self.embedding_per_data.extend(embedding.view(output_data.shape[0], -1).cpu().detach())
        self.output_per_data.extend(output_data.cpu().detach())
        for metric_func_name, metric_func in self.metric_funcs.items():
            if metric_func_name in ['AUROC', 'F1']:
                pass
            else:
                metric_value = metric_func(batch_data, output_data)
                self.metrics_per_batch[metric_func_name] = metric_value
                self.test_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.test_data_loader)-1:
            if "AUROC" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['AUROC'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data))
                self.metrics_per_batch['AUROC'] = metric_value
                self.test_metrics_per_epoch['AUROC'].update(metric_value)
            if "F1" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['F1'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data), self.args.anomaly_threshold)
                self.metrics_per_batch['F1'] = metric_value
                self.test_metrics_per_epoch['F1'].update(metric_value)
            if self.args.save_embedding:
                self.tensorboard_writer_test.add_embedding(torch.stack(self.embedding_per_data), self.labels_per_data, torch.stack(self.output_per_data), int(self.epoch_idx), 'embedding_vector')
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch)
        if self.args.save_result_images:
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, batch_data, batch_label, 'input')
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, output_data, batch_label, 'output')

    def _set_testing_variables(self):
        super()._set_testing_variables()
        for loss_name in self.model.get_losses_name():
            self.test_losses_per_epoch[loss_name] = AverageMeter()
        self.embedding_per_data = []
        self.output_per_data = []