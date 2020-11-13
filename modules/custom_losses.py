import pytorch_msssim
from torchvision.transforms.functional import normalize


# class Losses(object):
#     def __init__(self):
#         pass

#     def __call__(self):
#         try:
#             pass
#         except Exception as e:
#             print(e)

class SSIM():
    def __init__(self, normalize, channel_num):
        if normalize:
            self.ssim_module = pytorch_msssim.SSIM(data_range=1, size_average=False, channel=1)
        else:
            self.ssim_module = pytorch_msssim.SSIM(data_range=255, size_average=False, channel=1)
        self.channel_num = channel_num

    def __call__(self, X, Y):
        try:
            if self.channel_num != 1:
                X = ((0.2989*X[:,0] + 0.5870*X[:,1] + 0.1140*X[:,2]).unsqueeze(1))
                Y = ((0.2989*Y[:,0] + 0.5870*Y[:,1] + 0.1140*Y[:,2]).unsqueeze(1))
            ssim = self.ssim_module(X,Y)
            return (1.0-ssim)
        except Exception as e:
            print(e)