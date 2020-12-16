from PIL import Image
import numpy as np
import os
root = '/root/anomaly_detection/temp'
img_files_name = os.listdir(root)
gt_images = []
output_images = []
reses = []
for img_filename in img_files_name:
    filepath = os.path.join(root, img_filename)
    img = Image.open(filepath)
    if 'gt' in filepath:
        gt_images.append([filepath, img])
    elif 'output' in filepath:
        output_images.append([filepath, img])
output_images = sorted(output_images)
gt_images = sorted(gt_images)
for i in range(len(gt_images)):
    res = np.asarray(output_images[i][1])/255 - np.asarray(gt_images[i][1])/255
    res = np.mean(np.abs(res))
    reses.append([output_images[i][0], res])
print('Finished')
# img = cv2.imread('/)