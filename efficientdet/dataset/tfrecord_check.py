'''
TFRecord에서 체크해야 하는 부분 정리 
- class id, 이름 제대로 안들어가 있는 이미지가 몇개나 있는지 확인
    - class 이름 체크
    - class id 체크
- 이미지 경로 잘 들어가나 확인 → 아니면 변환할때 오류 발생
- train data statistic 알려주기
'''
import tensorflow as tf 
import os 
LABELMAP= {1: "CL", 2: "CC", 3: "SD", 4: "BK", 5: "LP", 6: "JF", 7: "JD", 8: "DS", 9: "ETC", 10: "PJ", 11: "IN", 12: "CAR", 13: "INV", 14: "HL"}

# 반복문 구현 


root = "/hd/sewer/sewer_100/tfrecord/val/"
num_shard = 50
# for datatype in ['train', 'val']:
PREFIX_LIST = [value for key, value in LABELMAP.items()]
for prefix in PREFIX_LIST:
    num_data_list = []
    num_annotation_data_list = []
    for index in [0, 1]:#range(num_shard):
        num_data = 0
        num_annotation_data = 0
        filenames = os.path.join(root, "%s-%05d-of-000%d.tfrecord"%(prefix, index, num_shard))
        if not os.path.isfile(filenames):
            continue
        raw_dataset = tf.data.TFRecordDataset(filenames)
        for raw_record in raw_dataset:
            num_data += 1
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            img_filename = example.features.feature['image/filename'].bytes_list.value
            bbox_xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
            bbox_xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
            bbox_ymin = example.features.feature['image/object/bbox/ymin'].float_list.value
            bbox_ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
            cls_name = example.features.feature['image/object/class/text'].bytes_list.value
            cls_label = example.features.feature['image/object/class/label'].int64_list.value
            # print('xmin', bbox_xmin, 'xmax', bbox_xmax)
            # print('ymin', bbox_ymin, 'ymax', bbox_ymax)
            # print('cls name', cls_name, 'cls label', cls_label)
            if not len(cls_name) == 0:
                num_annotation_data += 1
                print('img_filename', img_filename)
                print('xmin', bbox_xmin, 'xmax', bbox_xmax)
                print('ymin', bbox_ymin, 'ymax', bbox_ymax)
                print('cls name', cls_name, 'cls label', cls_label)
                for label, name  in zip(cls_label, cls_name):
                    if not bytes(LABELMAP[label], "utf8") == name:
                        print("WRONG CLS NAME!")
                

        # print('FILENAMES : %s'%(filenames))
        # print('TOTAL : %d, ANNOTATED : %d'%(num_data, num_annotation_data))
        num_data_list.append(num_data)
        num_annotation_data_list.append(num_annotation_data)
    print('CLS : %s, TOTAL : %d, ANNOTATED : %d'%(prefix, sum(num_data_list), sum(num_annotation_data_list)))
