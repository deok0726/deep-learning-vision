import xml.etree.cElementTree as ET
import os.path as osp
import os
import sys
# import numpy as np

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


xml_root = '/hd/MOT/DETRAC/XML'
label_root = '/hd/MOT/DETRAC/labels_with_ids'
mkdirs(label_root)
seqs = [s[:9] for s in os.listdir(xml_root)]
seqs.sort()

for seq in seqs:
    mkdirs(osp.join(label_root, seq))
    seq_width = 960
    seq_height = 540
    tree = ET.parse(osp.join(xml_root, seq + '.xml'))
    root = tree.getroot()
    
    # for child in root:
    #     if child.tag != 'frame':
    #         continue
    #     print("img%05d"%(int(child.attrib.get("num"))))

    tid_curr = 0
    tid_last = -1
    for frame in root.findall('frame'):
        label_fpath = osp.join(label_root, seq, "img%05d.txt"%(int(frame.get("num"))))
        # print(label_fpath)
        for target_list in frame.findall('target_list'):
            for target in target_list.findall('target'):
                # print('target id is ' + target.get('id'))
                tid = int(target.get('id'))
                attribute = target.find('attribute')
                # label = attribute.get('vehicle_type')
                # if label != 'car':
                #     continue

                box = target.find('box')
                x = float(box.get('left'))
                y = float(box.get('top'))
                w = float(box.get('width'))
                h = float(box.get('height'))

                x += w / 2
                y += h / 2

                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)

                # print(label)
                # print(label_str)