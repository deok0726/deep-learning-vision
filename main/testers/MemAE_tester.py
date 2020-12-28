from modules.utils import AverageMeter
from modules.custom_metrics import classification_report, confusion_matrix, rapp_criterion
from main.testers.tester import Tester
import time
import torch
import numpy as np

class MemAETester(Tester):
    def __init__(self, args, dataloader, model, optimizer, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, loss_funcs, metric_funcs, device)
        # self.ANOMALY_CRITERION = rapp_criterion
        self.window_size = args.window_size
    
    def _get_valid_residuals(self, batch_data):
        batch_data = batch_data.to(self.device)
        with torch.no_grad():
            if self.args.random_crop:
                output_data = self._generate_patches_and_merge(batch_data, self.window_size, self.args.crop_size, self.args.channel_num)[0]
            else:
                output_data = self.model(batch_data)
                output_data = getattr(output_data ,'output')
        # batch_diff_per_batch = self.ANOMALY_CRITERION(batch_data, output_data, self.model.encoder.children())
        # batch_diff_per_batch = (batch_data - output_data) ** 2
        batch_diff_per_batch = torch.abs(batch_data - output_data)
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        if max(batch_diff_per_batch) > self.max_residual:
            self.max_residual = max(batch_diff_per_batch)
    
    def _get_diffs_per_data_threshold(self, batch_data, batch_label):
        batch_data = batch_data.to(self.device)
        with torch.no_grad():
            if self.args.random_crop:
                output_data = self._generate_patches_and_merge(batch_data, self.window_size, self.args.crop_size, self.args.channel_num)[0]
            else:
                output_data = self.model(batch_data)
                output_data = getattr(output_data ,'output')
        # batch_diff_per_batch = self.ANOMALY_CRITERION(batch_data, output_data, self.model.encoder.children())
        # batch_diff_per_batch = (batch_data - output_data) ** 2
        batch_diff_per_batch = torch.abs(batch_data - output_data)
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        self.diffs_per_data_threshold.extend(batch_diff_per_batch.cpu().detach().numpy())
        self.labels_per_data_threshold.extend(batch_label.cpu().detach().numpy())

    def _test_step(self, batch_data, batch_label):
        self.data_time.update(time.time() - self.end_time)
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        with torch.no_grad():
            if self.args.random_crop:
                output_data = self._generate_patches_and_merge(batch_data, self.window_size, self.args.crop_size, self.args.channel_num)
            else:
                output_data = self.model(batch_data)
                output_data = getattr(output_data ,'output')
        # batch_diff_per_batch = (batch_data - output_data) ** 2
        batch_diff_per_batch = torch.abs(batch_data - output_data)
        batch_diff_per_batch = batch_diff_per_batch.mean((1, 2, 3))
        self.diffs_per_data.extend(batch_diff_per_batch.cpu().detach().numpy())
        self.labels_per_data.extend(batch_label.cpu().detach().numpy())
        self.output_per_data.extend(output_data.cpu().detach())
        for metric_func_name, metric_func in self.metric_funcs.items():
            if metric_func_name in ['AUROC', 'AUPRC', 'F1', 'Recall']:
                pass
            else:
                metric_value = metric_func(batch_data, output_data)
                self.metrics_per_batch[metric_func_name] = metric_value
                self.test_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.args.save_result_images:
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, batch_data, batch_label, batch_diff_per_batch, 'input')
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, output_data, batch_label, batch_diff_per_batch, 'output')
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, torch.abs(batch_data-output_data), batch_label, batch_diff_per_batch, 'residual')
        if self.batch_idx == len(self.dataloader.test_data_loader)-1:
            if "AUROC" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['AUROC'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data))
                self.metrics_per_batch['AUROC'] = metric_value
                self.test_metrics_per_epoch['AUROC'].update(metric_value)
            if "AUPRC" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['AUPRC'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data))
                self.metrics_per_batch['AUPRC'] = metric_value
                self.test_metrics_per_epoch['AUPRC'].update(metric_value)
            if self.args.anomaly_threshold:
                print(confusion_matrix(np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data), self.args.anomaly_threshold, self.args.target_label, self.args.unique_anomaly))
                print(classification_report(np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data), self.args.anomaly_threshold, self.args.target_label, self.args.unique_anomaly))
                if "F1" in self.metric_funcs.keys():
                    metric_value = self.metric_funcs['F1'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data), self.args.anomaly_threshold)
                    self.metrics_per_batch['F1'] = metric_value
                    self.test_metrics_per_epoch['F1'].update(metric_value)
                if "Recall" in self.metric_funcs.keys():
                    metric_value = self.metric_funcs['Recall'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data), self.args.anomaly_threshold)
                    self.metrics_per_batch['Recall'] = metric_value
                    self.test_metrics_per_epoch['Recall'].update(metric_value)
            if self.args.save_embedding:
                self.tensorboard_writer_test.add_embedding(torch.stack(self.embedding_per_data), self.labels_per_data, torch.stack(self.output_per_data), int(self.epoch_idx), 'embedding_vector')
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch)

    def _set_testing_variables(self):
        super()._set_testing_variables()
        for loss_name in self.model.get_losses_name():
            self.test_losses_per_epoch[loss_name] = AverageMeter()
        self.embedding_per_data = []
        self.output_per_data = []
    
    def _generate_patches_and_merge(self, batch_data, stride, windowSize, output_color_channel):
        output_data = torch.empty((output_color_channel, batch_data.shape[2], batch_data.shape[3])).unsqueeze(0)
        for y in range(0, batch_data.shape[2], stride):
            for x in range(0, batch_data.shape[3], stride):
                if (y+windowSize > batch_data.shape[2]) or (x+windowSize > batch_data.shape[3]):
                    pass
                else:
                    batch_data_patch = batch_data[:,:, y:y+windowSize, x:x+windowSize]
                    output_data_patch = self.model(batch_data_patch)
                    output_data[:,:, y:y+windowSize, x:x+windowSize] = getattr(output_data_patch, 'output')[0]
        return output_data.cuda()