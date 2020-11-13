import os
import matplotlib.pyplot as plt
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def automkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#         print(path+' generated')
#     else:
#         print(path+' exists')

def matplotlib_imshow(img, one_channel=False, normalized=False, mean=0.5, std=0.5):
    if normalized:
        img = img.mul(std).add(mean)
    img = img.cpu()
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="gray") # or Greys
    else:
        plt.imshow(np.transpose(np.clip(npimg, 0, 1), (1, 2, 0)))
        # plt.imshow(np.transpose(((np.clip(npimg, 0, 1)*255).astype('uint8')), (1, 2, 0)))