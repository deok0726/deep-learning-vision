from PIL import Image
import os
from glob import glob
import sys

DATA_PATH = '/hd/daejoo/amber/test/defective/'

def rgba2rgb():
    fpattern = os.path.join(DATA_PATH, '*.png')
    fpaths = sorted(glob(fpattern))

    for i in range(len(fpaths)):
        rgba_image = Image.open(fpaths[i])
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save(os.path.join('/hd/daejoo/amber/test/defective/', os.path.basename(fpaths[i])))


if __name__ == '__main__':
    rgba2rgb()