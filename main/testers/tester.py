import os
import torch
import cv2
from torch.utils import tensorboard
import time, datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import ceil
from modules.utils import AverageMeter, matplotlib_imshow
from modules.custom_metrics import classification_report, confusion_matrix

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
        # method 1 - using only validation dataset
        # for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.valid_data_loader), total=len(self.dataloader.valid_data_loader), desc='Valid'):
        #     self._get_valid_residuals(batch_data)
        # if self.max_residual != 0:
        #     self.args.anomaly_threshold = self.max_residual.item()
        # method 2 - using small test dataset
        for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.test_threshold_data_loader), total=len(self.dataloader.test_threshold_data_loader), desc='Test_Threshold'):
            self._get_diffs_per_data_threshold(batch_data, batch_label)
        self.args.anomaly_threshold = self._get_threshold(self.diffs_per_data_threshold, self.labels_per_data_threshold, self.thresholds_candidates, [self.metric_funcs['Recall'], self.metric_funcs['F1']])
        print("anomaly_threshold: ", self.args.anomaly_threshold)
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
        if self.batch_idx == (len(self.dataloader.test_data_loader)-1):
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
            self._log_tensorboard(batch_data, batch_label, output_data, self.losses_per_batch, self.metrics_per_batch)

    def _get_valid_residuals(self, batch_data):
        return None
    
    def _get_diffs_per_data_threshold(self, batch_data, batch_label):
        return None

    def _get_threshold(self, diffs, labels, threshold_candidates, scoring_funcs):
        scores_all = []
        for thresholds_candidate in threshold_candidates:
            scores = []
            for scoring_func in scoring_funcs:
                score = scoring_func(np.asarray(diffs), np.asarray(labels), thresholds_candidate)
                scores.append(score)
            scores_all.append(scores)
        scores_all = np.array(scores_all) #99, 2
        max_recall = scores_all[scores_all[:, 0].argmax(), 0]
        max_recall_idxes = np.where(scores_all[:, 0] == max_recall)
        scores_F = scores_all[max_recall_idxes, 1]
        threshold_candidates_F = threshold_candidates[max_recall_idxes]
        threshold = threshold_candidates_F[scores_F.argmax()]
        return threshold

    def _set_testing_constants(self):
        self.CHECKPOINT_SAVE_DIR = os.path.join(os.path.join(self.args.checkpoint_dir, self.args.model_name), self.args.exp_name)
        self.TENSORBOARD_LOG_SAVE_DIR = os.path.join(os.path.join(self.args.tensorboard_dir, self.args.model_name), self.args.exp_name)
        self.TEST_RESULTS_SAVE_DIR = os.path.join(self.TENSORBOARD_LOG_SAVE_DIR, 'test_results')
        self.TIMEZONE = datetime.timezone(datetime.timedelta(hours=9))
        self.TEST_LOG_INTERVAL = ceil(len(self.dataloader.test_data_loader) / 10)
        self.ANOMALY_CRITERION = torch.nn.L1Loss(reduction='none')
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
        self.diffs_per_data_threshold = []
        self.labels_per_data_threshold = []
        # self.valid_residuals = []
        self.max_residual = 0
        self.thresholds_candidates = np.arange(0.0001, 1, 0.0002)
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
        fig = plt.figure(figsize=(8, 12))
        for idx in range(self.args.test_tensorboard_shown_image_num):
            losses = []
            metrics = []
            random_sample_idx = np.random.randint(0, output_data.shape[0])
            ax_batch = fig.add_subplot(3, self.args.test_tensorboard_shown_image_num, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(batch_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_batch.set_title("Ground Truth")
            ax_output = fig.add_subplot(3, self.args.test_tensorboard_shown_image_num, idx+self.args.test_tensorboard_shown_image_num+1, xticks=[], yticks=[])
            matplotlib_imshow(output_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            for loss_per_batch_name, loss_per_batch_value in losses_per_batch.items():
                losses.append(':'.join((loss_per_batch_name, str(round(loss_per_batch_value[random_sample_idx].mean().item(), 5)))))
            for metric_per_batch_name, metric_per_batch_value in metrics_per_batch.items():
                if metric_per_batch_name in ['AUROC', 'AUPRC', 'F1', 'Recall']:
                    pass
                else:
                    metrics.append(':'.join((metric_per_batch_name, str(round(metric_per_batch_value[random_sample_idx].mean().item(), 5)))))
            if 'AUROC' in metric_per_epoch.keys():
                metrics.append(':'.join(('AUROC', str(round(metric_per_epoch['AUROC'].avg, 5)))))
            if 'AUPRC' in metric_per_epoch.keys():
                metrics.append(':'.join(('AUPRC', str(round(metric_per_epoch['AUPRC'].avg, 5)))))
            if 'F1' in metric_per_epoch.keys():
                metrics.append(':'.join(('F1', str(round(metric_per_epoch['F1'].avg, 5)))))
            if 'Recall' in metric_per_epoch.keys():
                metrics.append(':'.join(('Recall', str(round(metric_per_epoch['Recall'].avg, 5)))))
            ax_output.set_title("Output\n" + "losses\n" + "\n".join(losses) + "\n\nmetrics\n"+ "\n".join(metrics) + "\nlabel: " + str(batch_label[random_sample_idx].item()))
            ax_residual = fig.add_subplot(3, self.args.test_tensorboard_shown_image_num, idx+self.args.test_tensorboard_shown_image_num*2+1, xticks=[], yticks=[])
            matplotlib_imshow(torch.abs(batch_data[random_sample_idx]-output_data[random_sample_idx]) , one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_residual.set_title("Residual Map")
            
        plt.tight_layout()
        self.tensorboard_writer_test.add_figure("test", fig, global_step=self.epoch_idx)
        for loss_name, loss_value_per_epoch in losses_per_epoch.items():
            scalar_tag = [loss_name, '/loss']
            self.tensorboard_writer_test.add_scalar(''.join(scalar_tag), loss_value_per_epoch.avg, self.epoch_idx)
        for metric_name, metric_value_per_epoch in metric_per_epoch.items():
            scalar_tag = [metric_name, '/metric']
            self.tensorboard_writer_test.add_scalar(''.join(scalar_tag), metric_value_per_epoch.avg, self.epoch_idx)
        # self.tensorboard_writer_test.add_pr_curve('test_pr_curve', np.asarray(self.labels_per_data), np.asarray(self.diffs_per_data), self.epoch_idx)
        self.log_pr_curve(np.asarray(self.labels_per_data), np.asarray(self.diffs_per_data), self.epoch_idx)
        self.tensorboard_writer_test.flush()
    
    def log_pr_curve(self, labels, scores, step):
        if(self.args.unique_anomaly):
            labels[labels == self.args.target_label] = -1
            labels[labels != -1] = 1
        else:
            labels[labels != self.args.target_label] = -1
            labels[labels != -1] = 1
        labels[labels==1] = 0
        labels[labels==-1] = 1
        scores = (scores - scores.min(axis=0)) / (scores.max(axis=0) - scores.min(axis=0))
        self.tensorboard_writer_test.add_pr_curve('test_pr_curve', labels, scores, step)

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
    
    def save_result_images(self, save_path, result_images, result_label, diff, result_type=None):
        for img_idx in range(result_label.shape[0]):
            image_path = os.path.join(save_path, str(self.batch_idx)+ '_' +str(img_idx))
            image_path = image_path + '_label_' + str(result_label[img_idx].item()) + '_' + result_type
            image_path = image_path + '_' + 'Anomaly_Criterion' + '_' + str(round(diff[img_idx].item(), 5))
            for loss_name, loss_value in self.losses_per_batch.items():
                image_path = image_path + '_' + loss_name + '_' + str(round(loss_value[img_idx].mean().item(), 5))
            for metric_name, metric_value in self.metrics_per_batch.items():
                image_path = image_path + '_' + metric_name + '_' + str(round(metric_value[img_idx].mean().item(), 5))
            image_path += '.png'
            if self.args.normalize:
                result_img = result_images[img_idx].detach().mul(0.5).add(0.5)
            else:
                result_img = result_images[img_idx].detach()
            result_img = np.clip(result_img.cpu().numpy().transpose(1, 2, 0), 0, 1)
            result_img = (result_img*255).astype('uint8')
            cv2.imwrite(image_path, result_img[..., ::-1])