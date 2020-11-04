from modules.utils import AverageMeter, matplotlib_imshow
from main.testers.tester import Tester
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.utils import save_image
from itertools import combinations, product

class ARNetTester(Tester):
    def __init__(self, args, dataloader, model, optimizer, loss_funcs: dict, metric_funcs: dict, device):
        super().__init__(args, dataloader, model, optimizer, loss_funcs, metric_funcs, device)
        self.transformation_functions = dict(
            rotate = dict(
                rotate_0 = lambda x: x,
                rotate_90 = lambda x: x.transpose(-2, -1).flip(-2),
                rotate_180 = lambda x: x.flip(-2).flip(-1),
                rotate_270 = lambda x: x.transpose(-2, -1).flip(-1),
            ),
            hflips = dict(
                no_hflip = lambda x: x,
                hflip = torchvision.transforms.functional.hflip
            ),
            vflips = dict(
                no_vflip = lambda x: x,
                vflip = torchvision.transforms.functional.vflip
            )
        )
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
            if (batch_idx % self.TEST_LOG_INTERVAL == 0) or (batch_idx + 1 == len(self.dataloader.test_data_loader)):
                print('Test Batch Step', 'batch idx', self.batch_idx, 'batch data shape', batch_data.shape)
                self._log_progress()
        self.tensorboard_writer_test.close()
    
    def _get_expectation_error_step(self, input_batch_data):
        original_batch_data = []
        transformed_batch_data_all = []
        all_types_products = []
        #TBD: speed up
        # Graying: This operation averages each pixel value along the channel dimension of images.
        if input_batch_data.shape[1] == 3:
            grayed_batch_data = torch.mean(input_batch_data, dim=1, keepdim=True) # channel dim 2
        
        # original_batch_data = input_batch_data
        # transformed_batch_data_all = grayed_batch_data
        # b, c, h, w = transformed_batch_data_all.shape
        # t = 1
        
        # Random rotation: This operation rotates x anticlockwise by angle alpha around the center of each image channel. The rotation angle alpha is randomly selected from a set {0, 90, 180, 270}
        # hflip
        # vflip
        for k in range(len(self.transformation_functions.keys())):
            transformation_combinations = list(combinations(self.transformation_functions.keys(), k+1)) 
            for transformation_combination in transformation_combinations: # ex) ['rotate', 'hflips']
                all_types = []
                for transformation_function in transformation_combination:
                    all_types.append(list(self.transformation_functions[transformation_function].keys()))
                types_products = list(product(*all_types))
                for types_product in types_products:
                    all_types_products.append(types_product)
                    for idx, transformation_function in enumerate(transformation_combination):
                        if idx == 0:
                            transformed_batch_data = self.transformation_functions[transformation_function][types_product[idx]](grayed_batch_data)
                        else:
                            transformed_batch_data = self.transformation_functions[transformation_function][types_product[idx]](transformed_batch_data)
                    transformed_batch_data_all.append(transformed_batch_data)
                    original_batch_data.append(input_batch_data)
        original_batch_data = torch.stack(original_batch_data, dim=0) # num of transformations, B, C, H, W
        transformed_batch_data_all = torch.stack(transformed_batch_data_all, dim=0) # num of transformations, B, C, H, W
        original_batch_data = torch.transpose(original_batch_data, 0, 1) # num of transformations, B
        transformed_batch_data_all = torch.transpose(transformed_batch_data_all, 0, 1) # B, num of transformations, C, H, W
        if not original_batch_data.is_contiguous():
            original_batch_data = original_batch_data.contiguous()
        if not transformed_batch_data_all.is_contiguous():
            transformed_batch_data_all = transformed_batch_data_all.contiguous()
        b, t, c, h, w = transformed_batch_data_all.shape
        original_batch_data = original_batch_data.flatten(0, 1)
        transformed_batch_data_all = transformed_batch_data_all.flatten(0, 1)

        original_batch_data = original_batch_data.to(self.device)
        transformed_batch_data_all = transformed_batch_data_all.to(self.device)
        # output_data = []
        output_data_all = []
        with torch.no_grad():
            for chunk_idx in range(b*t):
                # sliding window
                # original_batch_data_patch, transformed_batch_data_all_patch, output_data_patch, labels_patch = self._generate_patches(
                #     original_batch_data[chunk_idx],
                #     transformed_batch_data_all[chunk_idx],
                #     original_batch_label[chunk_idx],
                #     30,
                #     self.args.crop_size)
                output_data = self._generate_patches_and_merge(transformed_batch_data_all[chunk_idx], 100, self.args.crop_size, self.args.channel_num)
                # original_batch_data_patches.append(original_batch_data_patch)
                # transformed_batch_data_all_patches.append(transformed_batch_data_all_patch)
                output_data_all.append(output_data)
                # original_batch_label_patches.append(labels_patch)
                
                # output_data.append(self.model(transformed_batch_data_all[chunk_idx].unsqueeze(0)))
        
        # original_batch_data = torch.stack(original_batch_data_patches, dim=0)
        # transformed_batch_data_all = torch.stack(transformed_batch_data_all_patches, dim=0)
        # output_data = torch.stack(output_data_patches, dim=0)
        # original_batch_label = torch.stack(original_batch_label_patches, dim=0)

        output_data = torch.stack(output_data_all, dim=0)
        output_data = output_data.to(self.device)
        # with torch.no_grad():
        #     for chunk_idx in range(b*t):
        #         output_data.append(self.model(transformed_batch_data_all[chunk_idx].unsqueeze(0)))
        # output_data = torch.stack(output_data, dim=0).squeeze()
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
        transformed_batch_data_all = []
        original_batch_label = []
        all_types_products = []
        #TBD: speed up
        # Graying: This operation averages each pixel value along the channel dimension of images.
        if input_batch_data.shape[1] == 3:
            grayed_batch_data = torch.mean(input_batch_data, dim=1, keepdim=True) # channel dim 2
        
        # original_batch_data = input_batch_data
        # transformed_batch_data_all = grayed_batch_data
        # original_batch_label = input_batch_label
        # b, c, h, w = transformed_batch_data_all.shape
        # t = 1

        # Random rotation: This operation rotates x anticlockwise by angle alpha around the center of each image channel. The rotation angle alpha is randomly selected from a set {0, 90, 180, 270}
        # hflip
        # vflip
        for k in range(len(self.transformation_functions.keys())):
            transformation_combinations = list(combinations(self.transformation_functions.keys(), k+1)) 
            for transformation_combination in transformation_combinations: # ex) ['rotate', 'hflips']
                all_types = []
                for transformation_function in transformation_combination:
                    all_types.append(list(self.transformation_functions[transformation_function].keys()))
                types_products = list(product(*all_types))
                for types_product in types_products:
                    all_types_products.append(types_product)
                    for idx, transformation_function in enumerate(transformation_combination):
                        if idx == 0:
                            transformed_batch_data = self.transformation_functions[transformation_function][types_product[idx]](grayed_batch_data)
                        else:
                            transformed_batch_data = self.transformation_functions[transformation_function][types_product[idx]](transformed_batch_data)
                    transformed_batch_data_all.append(transformed_batch_data)
                    original_batch_data.append(input_batch_data)
                    original_batch_label.append(input_batch_label)
        transformed_batch_data_all = torch.stack(transformed_batch_data_all, dim=0) # num of angles, B, C, H, W
        original_batch_data = torch.stack(original_batch_data, dim=0) # num of angles, B, C, H, W
        original_batch_label = torch.stack(original_batch_label, dim=0) # num of angles, B
        transformed_batch_data_all = transformed_batch_data_all.transpose(0, 1)
        original_batch_data = original_batch_data.transpose(0, 1)
        original_batch_label = original_batch_label.transpose(0, 1)
        if not original_batch_data.is_contiguous():
            original_batch_data = original_batch_data.contiguous()
        if not transformed_batch_data_all.is_contiguous():
            transformed_batch_data_all = transformed_batch_data_all.contiguous()
        if not original_batch_label.is_contiguous():
            original_batch_label = original_batch_label.contiguous()
        b, t, c, h, w = transformed_batch_data_all.shape
        transformed_batch_data_all = transformed_batch_data_all.flatten(0, 1)
        original_batch_data = original_batch_data.flatten(0, 1)
        original_batch_label = original_batch_label.flatten(0, 1)
        
        original_batch_data = original_batch_data.to(self.device)
        transformed_batch_data_all = transformed_batch_data_all.to(self.device)
        original_batch_label = original_batch_label.to(self.device)
        # original_batch_data_patches = []
        # transformed_batch_data_all_patches = []
        # original_batch_label_patches = []
        # output_data_patches = []
        output_data_all = []
        with torch.no_grad():
            for chunk_idx in range(b*t):
                # sliding window
                # original_batch_data_patch, transformed_batch_data_all_patch, output_data_patch, labels_patch = self._generate_patches(
                #     original_batch_data[chunk_idx],
                #     transformed_batch_data_all[chunk_idx],
                #     original_batch_label[chunk_idx],
                #     30,
                #     self.args.crop_size)
                output_data = self._generate_patches_and_merge(transformed_batch_data_all[chunk_idx], 100, self.args.crop_size, self.args.channel_num)
                # original_batch_data_patches.append(original_batch_data_patch)
                # transformed_batch_data_all_patches.append(transformed_batch_data_all_patch)
                output_data_all.append(output_data)
                # original_batch_label_patches.append(labels_patch)
                
                # output_data.append(self.model(transformed_batch_data_all[chunk_idx].unsqueeze(0)))
        
        # original_batch_data = torch.stack(original_batch_data_patches, dim=0)
        # transformed_batch_data_all = torch.stack(transformed_batch_data_all_patches, dim=0)
        # output_data = torch.stack(output_data_patches, dim=0)
        # original_batch_label = torch.stack(original_batch_label_patches, dim=0)

        output_data = torch.stack(output_data_all, dim=0)
        output_data = output_data.to(self.device)
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
            self._log_tensorboard(original_batch_data, original_batch_label, transformed_batch_data_all, output_data, self.losses_per_batch, self.metrics_per_batch)
        if self.args.save_result_images:
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, transformed_batch_data_all, original_batch_label, 'input')
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, output_data, original_batch_label, 'output')
            self.save_result_images(self.TEST_RESULTS_SAVE_DIR, original_batch_data, original_batch_label, 'gt')

    def _generate_patches_and_merge(self, transformed_data, stride, windowSize, output_color_channel):
        output_data = torch.empty((output_color_channel, transformed_data.shape[1], transformed_data.shape[2]))
        for y in range(0, transformed_data.shape[1], stride):
            for x in range(0, transformed_data.shape[2], stride):
                if (y+windowSize > transformed_data.shape[1]) or (x+windowSize > transformed_data.shape[2]):
                    pass
                else:
                    transformed_data_patch = transformed_data[:, y:y+windowSize, x:x+windowSize]
                    output_data_patch = self.model(transformed_data_patch.unsqueeze(0))
                    output_data[:, y:y+windowSize, x:x+windowSize] = output_data_patch
        return output_data

    # def _generate_patches(self, input_data, transformed_data, label, stride, windowSize):
    #     input_data_patches = []
    #     transformed_data_patches = []
    #     output_data_patches = []
    #     labels = []
    #     for y in range(0, input_data.shape[1], stride):
    #         for x in range(0, input_data.shape[2], stride):
    #             if (y+windowSize > input_data.shape[1]) or (x+windowSize > input_data.shape[2]):
    #                 pass
    #             else:
    #                 input_data_patch = input_data[:, y:y+windowSize, x:x+windowSize]
    #                 input_data_patches.append(input_data_patch)
    #                 transformed_data_patch = transformed_data[:, y:y+windowSize, x:x+windowSize]
    #                 transformed_data_patches.append(transformed_data_patch)
    #                 output_data_patch = self.model(transformed_data_patch.unsqueeze(0))
    #                 output_data_patches.append(output_data_patch)
    #                 labels.append(label)
    #     input_data_patches = torch.stack(input_data_patches, dim=0)
    #     transformed_data_patches = torch.stack(transformed_data_patches, dim=0)
    #     output_data_patches = torch.stack(output_data_patches, dim=0).squeeze(1)
    #     labels = torch.stack(labels, dim=0)
    #     return input_data_patches, transformed_data_patches, output_data_patches, labels

    def _set_testing_variables(self):
        super()._set_testing_variables()
        for loss_name in self.model.get_losses_name():
            self.test_losses_per_epoch[loss_name] = AverageMeter()
    
    def _log_tensorboard(self, batch_data, batch_label, transformed_batch_data_all, output_data, losses_per_batch, metrics_per_batch):
        losses_per_epoch = self.test_losses_per_epoch
        metric_per_epoch = self.test_metrics_per_epoch
        fig = plt.figure(figsize=(8, 12))
        for idx in range(self.args.test_tensorboard_shown_image_num):
            losses = []
            metrics = []
            random_sample_idx = np.random.randint(0, output_data.shape[0])
            ax_output = fig.add_subplot(3, self.args.test_tensorboard_shown_image_num, idx+self.args.test_tensorboard_shown_image_num+1, xticks=[], yticks=[])
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
            ax_batch = fig.add_subplot(3, self.args.test_tensorboard_shown_image_num, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(batch_data[random_sample_idx], one_channel=self.one_channel, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_batch.set_title("Ground Truth")
            ax_transfomred_batch = fig.add_subplot(3, self.args.test_tensorboard_shown_image_num, idx+self.args.test_tensorboard_shown_image_num*2+1, xticks=[], yticks=[])
            matplotlib_imshow(transformed_batch_data_all[random_sample_idx], one_channel=True, normalized=self.args.normalize, mean=0.5, std=0.5)
            ax_transfomred_batch.set_title("Transformed Batch Data")
        plt.tight_layout()
        self.tensorboard_writer_test.add_figure("test", fig, global_step=self.epoch_idx)
        # log losses
        for loss_name, loss_value_per_epoch in losses_per_epoch.items():
            scalar_tag = [loss_name, '/loss']
            self.tensorboard_writer_test.add_scalar(''.join(scalar_tag), loss_value_per_epoch.avg, self.epoch_idx)
        # log metrics
        for metric_name, metric_value_per_epoch in metric_per_epoch.items():
            scalar_tag = [metric_name, '/metric']
            self.tensorboard_writer_test.add_scalar(''.join(scalar_tag), metric_value_per_epoch.avg, self.epoch_idx)
        self.tensorboard_writer_test.flush()