import os.path as osp
import os
import numpy as np

import sys

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/hd/MOT/KITTI/data_tracking_label_2/training/label_02'
img_root = '/hd/MOT/KITTI/data_tracking_image_2/training/image_02/'
label_root = '/hd/MOT/KITTI/labels_with_ids/training/image_02'
mkdirs(label_root)

seqs = [s for s in os.listdir(seq_root)]
seqs.sort()

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq = seq.replace(".txt", "")
    seq_info = open(osp.join(img_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq + '.txt')
    gt = np.genfromtxt(gt_txt, dtype=None, delimiter=' ', encoding=None)

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    for fid, tid, obj_type, truncation, occlusion, obs_angle, x1, y1, x2, y2, w, h, l, X, Y, Z, tracker in gt:
        if not obj_type == 'Car' and not obj_type == 'Van' and not obj_type == 'Truck':
            continue
        fid = int(fid)
        tid = int(tid)

        x = (x2 - x1)/2
        y = (y2 - y1)/2
        w = x2 - x1
        h = y2 - y1
        
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
        
    # sys.exit()


# self.frame      = frame
# self.track_id   = track_id
# self.obj_type   = obj_type
# self.truncation = truncation
# self.occlusion  = occlusion
# self.obs_angle  = obs_angle
# self.x1         = x1
# self.y1         = y1
# self.x2         = x2
# self.y2         = y2
# self.w          = w
# self.h          = h
# self.l          = l
# self.X          = X
# self.Y          = Y
# self.Z          = Z
# self.yaw        = yaw
# self.score      = score
# self.ignored    = False
# self.valid      = False
# self.tracker    = -1