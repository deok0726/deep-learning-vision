from modules.utils import AverageMeter
from main.testers.tester import Tester
import time
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

class ARNetTester(Tester):
    def __init__(self, args, dataloader, model, optimizer, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, loss_funcs, metric_funcs, device)
        self.angles = [0, 90, 180, 270]
        self.rotate_funtions = {
            '0': lambda x, d1, d2: x,
            '90': lambda x, d1, d2: x.transpose(d1, d2).flip(d1),
            '180': lambda x, d1, d2: x.flip(d1).flip(d2),
            '270': lambda x, d1, d2: x.transpose(d1, d2).flip(d2)
            }
        self.anomaly_criterion = torch.nn.L1Loss(reduction='none')
        self.transform_avg_diff = None
    
    def test(self):
        self._restore_checkpoint()
        self.model.eval()
        self.end_time = time.time()
        print(len(self.dataloader.test_data_loader))
        for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.train_data_loader), total=len(self.dataloader.train_data_loader), desc='Get expectation error in Train dataset'):
            self.batch_idx = batch_idx
            self._get_expectation_error_step(batch_data)
            if (batch_idx + 1 == len(self.dataloader.train_data_loader)):
                self.transform_avg_diff = (self.transform_avg_diff / (batch_idx + 1))
        for batch_idx, (batch_data, batch_label) in tqdm(enumerate(self.dataloader.test_data_loader), total=len(self.dataloader.test_data_loader), desc='Test'):
            self.batch_idx = batch_idx
            self._test_step(batch_data, batch_label)
            if batch_idx % self.TEST_LOG_INTERVAL == 0:
                print('Test Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                self._log_progress()
        self.tensorboard_writer_test.close()
    
    def _get_expectation_error_step(self, input_batch_data):
        original_batch_data = []
        transformed_batch_data = []
        #TBD: speed up
        # Random rotation: This operation rotates x anticlockwise by angle alpha around the center of each image channel. The rotation angle alpha is randomly selected from a set {0, 90, 180, 270}
        for angle in self.angles:
            original_batch_data.append(input_batch_data)
            transformed_batch_data.append(self.rotate_funtions[str(angle)](input_batch_data, -2, -1))
        original_batch_data = torch.stack(original_batch_data, dim=0) # num of angles, B, C, H, W
        transformed_batch_data = torch.stack(transformed_batch_data, dim=0) # num of angles, B, C, H, W
        original_batch_data = torch.transpose(original_batch_data, 0, 1) # num of angles, B
        transformed_batch_data = torch.transpose(transformed_batch_data, 0, 1) # B, num of angles, C, H, W
        if not original_batch_data.is_contiguous():
            original_batch_data = original_batch_data.contiguous()
        if not transformed_batch_data.is_contiguous():
            transformed_batch_data = transformed_batch_data.contiguous()
        # Graying: This operation averages each pixel value along the channel dimension of images.
        if transformed_batch_data.shape[2] == 3:
            transformed_batch_data = torch.mean(transformed_batch_data, dim=2, keepdim=True) # channel dim 2
        b, t, c, h, w = transformed_batch_data.shape
        original_batch_data = original_batch_data.flatten(0, 1)
        transformed_batch_data = transformed_batch_data.flatten(0, 1)
        original_batch_data = original_batch_data.to(self.device)
        transformed_batch_data = transformed_batch_data.to(self.device)
        with torch.no_grad():
            output_data = self.model(transformed_batch_data)
        batch_diff_per_batch = self.anomaly_criterion(original_batch_data, output_data)
        transform_indexes = [0]*t
        transform_avg_diff = torch.zeros(t)
        for i in range(t):
            transform_indexes[i] = list(range(i, b*t, t))
            transform_avg_diff[i] = batch_diff_per_batch[transform_indexes[i]].mean() # b*t, c, h, w
        if self.transform_avg_diff is None:
            self.transform_avg_diff = transform_avg_diff
        else:
            self.transform_avg_diff += transform_avg_diff

    def _test_step(self, input_batch_data, input_batch_label):
        self.data_time.update(time.time() - self.end_time)
        original_batch_data = []
        transformed_batch_data = []
        original_batch_label = []
        #TBD: speed up
        # Random rotation: This operation rotates x anticlockwise by angle alpha around the center of each image channel. The rotation angle alpha is randomly selected from a set {0, 90, 180, 270}
        for angle in self.angles:
            transformed_batch_data.append(self.rotate_funtions[str(angle)](input_batch_data, -2, -1))
            original_batch_data.append(input_batch_data)
            original_batch_label.append(input_batch_label)
        transformed_batch_data = torch.stack(transformed_batch_data, dim=0) # num of angles, B, C, H, W
        original_batch_data = torch.stack(original_batch_data, dim=0) # num of angles, B, C, H, W
        original_batch_label = torch.stack(original_batch_label, dim=0) # num of angles, B
        transformed_batch_data = transformed_batch_data.transpose(0, 1)
        original_batch_data = original_batch_data.transpose(0, 1)
        original_batch_label = original_batch_label.transpose(0, 1)
        # Graying: This operation averages each pixel value along the channel dimension of images.
        if transformed_batch_data.shape[2] == 3:
            transformed_batch_data = torch.mean(transformed_batch_data, dim=2, keepdim=True) # channel dim 2
        b, t, c, h, w = transformed_batch_data.shape
        transformed_batch_data = transformed_batch_data.flatten(0, 1)
        original_batch_data = original_batch_data.flatten(0, 1)
        original_batch_label = original_batch_label.flatten(0, 1)
        if not original_batch_data.is_contiguous():
            original_batch_data = original_batch_data.contiguous()
        if not transformed_batch_data.is_contiguous():
            transformed_batch_data = transformed_batch_data.contiguous()
        if not original_batch_label.is_contiguous():
            original_batch_label = original_batch_label.contiguous()
        original_batch_data = original_batch_data.to(self.device)
        transformed_batch_data = transformed_batch_data.to(self.device)
        original_batch_label = original_batch_label.to(self.device)
        with torch.no_grad():
            output_data = self.model(transformed_batch_data)
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_values = loss_func(original_batch_data, output_data)
            self.losses_per_batch[loss_func_name] = loss_values
            self.test_losses_per_epoch[loss_func_name].update(loss_values.mean().item())
        args_for_losses = {
            'x': original_batch_data,
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
        batch_diff_per_batch = self.anomaly_criterion(original_batch_data, output_data) # l1
        batch_diff_per_batch_avg = torch.zeros(b)
        transform_indexes = [0]*t
        for i in range(t):
            transform_indexes[i] = list(range(i, b*t, t))
            batch_diff_per_batch[transform_indexes[i]] = batch_diff_per_batch[transform_indexes[i]] / self.transform_avg_diff[i]
        for i in range(b):
            batch_diff_per_batch_avg[i] = batch_diff_per_batch[t*i:t*(i+1)].mean()
        
        # for i in range(len(batch_diff_per_batch_avg)):
        #     for j in range(t):
        #         # if original_batch_label[t*i+j].item() == 0:
        #         save_image(original_batch_data[t*i+j].double().mul_(0.5).add_(0.5), "/root/anomaly_detection/temp/" + str(i)+'_'+str(j)+ "_gt_" + str(original_batch_label[t*i+j].item()) +'_' + str(batch_diff_per_batch[t*i+j].double().mean().item())+".png", "PNG")
        #         save_image(output_data[t*i+j].double().mul_(0.5).add_(0.5), "/root/anomaly_detection/temp/" + str(i)+'_'+str(j) + "_output_" +str(original_batch_label[t*i+j].item()) + '_' + str(batch_diff_per_batch[t*i+j].double().mean().item())+".png", "PNG")
        
        self.diffs_per_data.extend(batch_diff_per_batch_avg.cpu().detach().numpy())
        self.labels_per_data.extend(input_batch_label.cpu().detach().numpy())
        for metric_func_name, metric_func in self.metric_funcs.items():
            if metric_func_name == 'ROC':
                pass
            else:
                metric_value = metric_func(original_batch_data, output_data)
                self.metrics_per_batch[metric_func_name] = metric_value
                self.test_metrics_per_epoch[metric_func_name].update(metric_value.mean().item())
        self.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()
        if self.batch_idx == len(self.dataloader.test_data_loader)-1:
            if "ROC" in self.metric_funcs.keys():
                metric_value = self.metric_funcs['ROC'](np.asarray(self.diffs_per_data), np.asarray(self.labels_per_data))
                self.metrics_per_batch['ROC'] = metric_value
                self.test_metrics_per_epoch['ROC'].update(metric_value)
            self._log_tensorboard(original_batch_data, original_batch_label, output_data, self.losses_per_batch, self.metrics_per_batch, True)

    def _set_testing_variables(self):
        super()._set_testing_variables()
        for loss_name in self.model.get_losses_name():
            self.test_losses_per_epoch[loss_name] = AverageMeter()