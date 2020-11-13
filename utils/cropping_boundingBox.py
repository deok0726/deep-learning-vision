from xml.etree import ElementTree as ET
import cv2
import os

def xml2pkl_unit(xml_filepath):
    pkl_dict_unit = {}
    xml_tree = ET.ElementTree()
    xml_root = xml_tree.parse(xml_filepath)

    image_filename = xml_root.findtext('filename')

    xml_size = xml_root.find('size')
    pkl_dict_unit['image_width'] = int(xml_size.findtext('width'))
    pkl_dict_unit['image_height'] = int(xml_size.findtext('height'))

    pkl_dict_unit['obj'] = []
    pkl_dict_unit['dont'] = []
    for xml_object in xml_root.findall('object'):
        obj = {}
        obj['text'] = xml_object.findtext('name')

        xml_bndbox = xml_object.find('bndbox')
        xmin = int(xml_bndbox.findtext('xmin'))
        ymin = int(xml_bndbox.findtext('ymin'))
        xmax = int(xml_bndbox.findtext('xmax'))
        ymax = int(xml_bndbox.findtext('ymax'))
        obj['bbox'] = dict(
            ymin = ymin,
            ymax = ymax,
            xmin = xmin,
            xmax = xmax
        )
        # obj['bbox'] = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        pkl_dict_unit['obj'].append(obj)
    return image_filename, pkl_dict_unit

def save_bbox_img(image_root_path, save_root_path, image_filename, image_info):
    image = cv2.imread(os.path.join(image_root_path, image_filename), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    image_filename = image_filename.split('.')[0]
    for idx, obj in enumerate(image_info['obj']):
        path = os.path.join(image_root_path, obj['text'])
        if not os.path.exists(path):
            os.makedirs(path)
        bbox = obj['bbox']
        cropped_img = image[bbox['ymin']: bbox['ymax'], bbox['xmin']: bbox['xmax']]
        cropped_img_filename = image_filename + '_' + str(idx) + '_' + obj['text']
        cv2.imwrite(os.path.join(image_root_path, obj['text'], cropped_img_filename)+'.png', cropped_img)

if __name__ == "__main__":
    # img_root = "/hd/anomaly_detection/GMSHightech_anomaly_detection_hole/good"
    # xml_root = "/hd/anomaly_detection/GMSHightech_anomaly_detection_hole/good"
    # save_root = "/hd/anomaly_detection/GMSHightech_anomaly_detection_hole/good"
    roots = [
        "/hd/anomaly_detection/GMS/normal/1",
        "/hd/anomaly_detection/GMS/normal/2",
        "/hd/anomaly_detection/GMS/normal/3",
        "/hd/anomaly_detection/GMS/normal/4",
        "/hd/anomaly_detection/GMS/normal/5",
        "/hd/anomaly_detection/GMS/normal/6",
        "/hd/anomaly_detection/GMS/normal/7",
        "/hd/anomaly_detection/GMS/ng/ng",
        # "/hd/anomaly_detection/GMSHightech_anomaly_detection_hole",
        # "/hd/anomaly_detection/GMSHightech_anomaly_detection_hole/good",
        # "/hd/anomaly_detection/GMSHightech_anomaly_detection_hole/defective"
    ]
    for root in roots:
        xml_filenames = os.listdir(root)
        xml_filenames = [xml_filename for xml_filename in xml_filenames if 'xml' in xml_filename]
        xml_filenames = sorted(xml_filenames)
        for xml_filename in xml_filenames:
            xml_filepath = os.path.join(root, xml_filename)
            img_name, img_info = xml2pkl_unit(xml_filepath)
            save_bbox_img(root, root, img_name, img_info)