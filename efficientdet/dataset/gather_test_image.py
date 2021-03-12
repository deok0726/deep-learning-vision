import tensorflow as tf 
import os 
import sys 
LABELMAP= {1: "CL", 2: "CC", 3: "SD", 4: "BK", 5: "LP", 6: "JF", 7: "JD", 8: "DS", 9: "ETC", 10: "PJ", 11: "IN", 12: "CAR", 13: "INV", 14: "HL"}
PATH_DICT = {"CL":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-1.균열-길이(Crack-Longitudinal,CL)/images",
            "CC":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-2.균열-원주(Crack-Circumferential,CC)/images",
            "SD":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-2.표면손상(Surface-Damage,SD)/images",
            "BK":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-3.파손(Broken-Pipe,BK)/images",
            "LP":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-4.연결관-돌출(Lateral-Protruding,LP)/images",
            "JF":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-5.이음부-손상(Joint-Faulty,JF)/images",
            "JD":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-6.이음부-단차(Joint-Displaced,JD)/images", 
            "DS":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-7.토사퇴적(Deposits-Silty,DS)/images",
            "ETC":"/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-8.기타결함(Etc.,ETC)/images",
            "PJ":"/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-1.이음부(Pipe-Joint,PJ)/images",
            "IN":"/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-2.하수관로_내부(Inside,IN)/images",
            "CAR":"/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-3.하수관로_외부_자동차/images",
            "INV":"/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-2.하수관로_외부_인버트/images",
            "HL":"/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-1.하수관로_외부_맨홀/images"}

root = "/hd/sewer/sewer_100/tfrecord/test"
num_shard = 50

PREFIX_LIST = [value for key, value in LABELMAP.items()]

for prefix in PREFIX_LIST:
    
    dir_path = os.path.join("/hd/sewer/sewer_100/test_images", prefix)
    if not os.path.isdir(dir_path):
        os.system("mkdir %s"%(dir_path))
    f = open(os.path.join(dir_path,"test_images_list_%s.txt"%(prefix)),"wt")
    num_data_list = []
    num_annotation_data_list = []
    for index in range(45, num_shard):
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
            cls_name = example.features.feature['image/object/class/text'].bytes_list.value
            if not len(cls_name) == 0:
                num_annotation_data += 1
                img_filename = img_filename[0].decode("utf-8")
                f.write(img_filename)
                f.write("\n")
                command = 'cp "%s" "%s"'%(os.path.join(PATH_DICT[prefix], img_filename), os.path.join("/hd/sewer/sewer_100/test_images/", prefix, img_filename))
                os.system(command)
        num_data_list.append(num_data)
        num_annotation_data_list.append(num_annotation_data)
    f.close()
    print('CLS : %s, TEST_DATA_NUM : %d'%(prefix, sum(num_data_list)))
