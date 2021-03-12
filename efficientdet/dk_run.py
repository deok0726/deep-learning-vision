import tensorflow as tf 
import os 

filenames = "/home/test_small/BK-00045-of-00050.tfrecord"
raw_dataset = tf.data.TFRecordDataset(filenames)
for idx, raw_record in enumerate(raw_dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    img_filename = example.features.feature['image/filename'].bytes_list.value
    img_source_id = example.features.feature['image/source_id'].bytes_list.value
    bbox_xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
    bbox_xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
    bbox_ymin = example.features.feature['image/object/bbox/ymin'].float_list.value
    bbox_ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
    cls_name = example.features.feature['image/object/class/text'].bytes_list.value
    cls_label = example.features.feature['image/object/class/label'].int64_list.value
    
    print('img_filename', img_filename)
    print('img_source_id', img_source_id)
    print('GT_xmin', bbox_xmin, 'GT_xmax', bbox_xmax)
    print('GT_ymin', bbox_ymin, 'GT_ymax', bbox_ymax)
    print('GT_cls name', cls_name)
    if idx > 10:
        break