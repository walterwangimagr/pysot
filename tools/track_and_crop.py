from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import multiprocessing as mp

import cv2
import torch
import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from PIL import Image
import requests
import pickle
import json 

torch.backends.cudnn.benchmark = True


def query_yolov5(img_path):
    server_base = "http://127.0.0.1:5800"
    img = Image.open(img_path)
    headers = {'Content-Type': 'application/octet-stream'}
    response = requests.post(f"{server_base}/infer",
                            data=pickle.dumps(img), headers=headers)
    results = response.json()
    return json.loads(results['results'])


def parse_result_json(results):
    """parse the json result from inference 

    Args:
        results (list of dict): list of detected objects 

    Returns:
        bboxs: list of bbox, xyxy normalized form
        cls_ids: list of class id 
        confidences: list of confidence
    """
    bboxs = []
    confidences = []
    cls_ids = []

    for result in results:
        xmin = result['xmin']
        ymin = result['ymin']
        xmax = result['xmax']
        ymax = result['ymax']
        confidence = result['confidence']
        cls_id = result['class']
        bbox = [xmin, ymin, xmax, ymax]
        bbox = normalized_bbox(bbox, 324)
        
        bboxs.append(bbox)
        confidences.append(confidence)
        cls_ids.append(cls_id)
    
    return cls_ids, bboxs, confidences


def read_img_yolo_label(img_path):
    """read labels file based on path of image file 
    if ~/images/xx/xx/xx.jpg then the labels file locate in 
    ~/auto-labels-yolov5/xx/xx/xx.txt

    Args:
        img_path : path of image file 

    Returns:
        bboxs: list of bbox, xyxy normalized form
        cls_ids: list of class id 
        confidences: list of confidence
    """
    label_path = re.sub("images", "auto-labels-yolov5", img_path)
    label_path = re.sub(".jp.+", ".txt", label_path)
    bboxs = []
    cls_ids = []
    confidences = []
    with open(label_path, "r") as f:
        for line in f:
            content = line.strip().split()
            content = list(map(float,content))
            if len(content) == 6:
                cls_id = content[0]
                x = content[1]
                y = content[2]
                w = content[3]
                h = content[4]
                xmin = x - w / 2
                ymin = y - h / 2
                confidence = content[5]
                bbox = [xmin, ymin, w, h]
                bbox = xywh_to_xyxy(bbox)
                bboxs.append(bbox)
                cls_ids.append(cls_id)
                confidences.append(confidence)
    return cls_ids, bboxs, confidences


def start_tracker():
    config_file = "/home/walter/nas_cv/walter_stuff/git/pysot/experiments/siammask_r50_l3/config.yaml"
    cfg.merge_from_file(config_file)
    model_path = "/home/walter/nas_cv/walter_stuff/git/pysot/experiments/siammask_r50_l3/model.pth"
    device = torch.device('cuda')
    model = ModelBuilder()
    model.load_state_dict(torch.load(model_path,
            map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    tracker = build_tracker(model)
    return tracker


def xyxy_to_xywh(xyxy_bbox):
    """
    xmin ymin xmax ymax to xmin ymin w h
    """
    xmin = xyxy_bbox[0]
    ymin = xyxy_bbox[1]
    xmax = xyxy_bbox[2]
    ymax = xyxy_bbox[3]
    w = xmax - xmin
    h = ymax - ymin 
    return [xmin, ymin, w, h]


def xywh_to_xyxy(xywh_bbox):
    """
    xmin ymin w h to xmin ymin xmax ymax 
    """
    xmin = int(xywh_bbox[0])
    ymin = int(xywh_bbox[1])
    w = int(xywh_bbox[2])
    h = int(xywh_bbox[3])
    xmax = int(xmin + w)
    ymax = int(ymin + h)
    return [xmin, ymin, xmax, ymax]

# write yolo label label, xywh normalized form 
def save_labels(label_save_dir, img_name, cls_id, bbox, confidence):
    """
    save xyxy bbox to xywh yolo format
    """
    os.makedirs(label_save_dir, exist_ok=True)
    label_name = re.sub(".jp.+", ".txt", img_name)
    savePath = os.path.join(label_save_dir, label_name)

    with open(savePath, 'w') as f:
        if bbox:
            n_bbox = normalized_bbox(bbox, 324)
            n_bbox = xyxy_to_xywh(n_bbox)
            str_to_save = f"{cls_id} {n_bbox[0]} {n_bbox[1]} {n_bbox[2]} {n_bbox[3]} {confidence}"
            f.write(str_to_save)


def normalized_bbox(bbox, img_sz):
    return list(map(lambda num: num / img_sz, bbox))


def scale_bbox(n_bbox, img_sz):
    bbox = list(map(lambda num: num * img_sz, n_bbox))
    return list(map(int, bbox))


def debug(imgs):
    current_image_index = 0
    cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
    while True:
        
        img = imgs[current_image_index]
        cv2.imshow('video', img)
        key = cv2.waitKey(0)
        
        if key == 27: 
            break
        elif key == 83: 
            current_image_index = (current_image_index + 1) % len(imgs)
        elif key == 81:  
            current_image_index = (current_image_index - 1) % len(imgs)

    cv2.destroyAllWindows()


def draw_bbox_confidence(img, bbox, confidence, color=(0,255,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    position = (100, 280)
    thickness = 2

    if bbox:
        bbox = list(map(int, bbox))
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        cv2.putText(img, str(confidence), position, font, font_scale, color, thickness)


def calculate_iou(box1, box2):
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2
    
    # Calculate the coordinates of the intersection rectangle
    xmin_intersection = max(box1_xmin, box2_xmin)
    ymin_intersection = max(box1_ymin, box2_ymin)
    xmax_intersection = min(box1_xmax, box2_xmax)
    ymax_intersection = min(box1_ymax, box2_ymax)
    
    # Calculate the area of intersection
    intersection_width = max(0, xmax_intersection - xmin_intersection + 1)
    intersection_height = max(0, ymax_intersection - ymin_intersection + 1)
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union
    area_box1 = (box1_xmax - box1_xmin + 1) * (box1_ymax - box1_ymin + 1)
    area_box2 = (box2_xmax - box2_xmin + 1) * (box2_ymax - box2_ymin + 1)
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area
    
    return iou


def bbox_reach_edge(bbox, edge=[10,10,314,314]):
    # bbox is xyxy and scaled 
    xmin, ymin, xmax, ymax = bbox
    edge_xmin, edge_ymin, edge_xmax, edge_ymax = edge
    return xmin <= edge_xmin or ymin <= edge_ymin or xmax >= edge_xmax or ymax >= edge_ymax


def bbox_within_region_of_interest(bbox, roi=[35,35,245,245]):
    xmin, ymin, xmax, ymax = bbox
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
    x_center = int((xmin + xmax) / 2)
    y_center = int((ymin + ymax) / 2)
    return roi_xmin <= x_center and x_center <= roi_xmax and roi_ymin <= y_center and y_center <= roi_ymax


def large_bbox(bbox, im_w=324, im_h=324):
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin 
    # 324 * 324 / 6 = 17496
    return w * h >= 17496


def get_frames(frames_dir):
    frames = list(sorted(glob.glob(f"{frames_dir}/*.jp*")))
    return frames


# for each folder, each camera
def run_per_folder(frames_dir, barcode, cam, save_dir):
    src_dir = os.path.join(frames_dir, barcode, cam)
    frames = get_frames(src_dir)
    dst_dir = os.path.join(save_dir, barcode, cam)
    os.makedirs(dst_dir, exist_ok=True)
    
    tracker_init = False 
    

    for frame in frames:
        img = cv2.imread(frame)
        bbox = []
        confidence = 0
        
        results = query_yolov5(frame)
        print(results)
        cls_ids, bboxs, confidences = parse_result_json(results)

        use_od = False
        use_tracker = False

        if len(bboxs) == 1:
            bbox = scale_bbox(bboxs[0], 324)
            confidence = confidences[0]
            if (bbox_within_region_of_interest(bbox) and not bbox_reach_edge(bbox) and confidence > 0.6) or (large_bbox(bbox) and confidence > 0.4):
                use_od = True
        
        if not use_od and tracker_init:
            use_tracker = True
        
        if use_od:
            xywh = xyxy_to_xywh(bbox)
            tracker = start_tracker()
            tracker.init(img, xywh)
            tracker_init = True
            confidence = confidences[0]
        
        if use_tracker:
            outputs = tracker.track(img)
            bbox = outputs['bbox']
            confidence = outputs['best_score']
            bbox = xywh_to_xyxy(bbox)
            if bbox_reach_edge(bbox) or confidence < 0.9:
                bbox = []
                tracker_init = False

        if not use_od and not use_tracker:
            bbox = []
            tracker_init = False

        if bbox:
            print(bbox)
            basename = os.path.basename(frame)
            dst = os.path.join(dst_dir, basename)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(dst, crop)






# barcodes = os.listdir(frame_dir)
# for barcode in barcodes:
#     cams = os.listdir(os.path.join(frame_dir, barcode))
#     for cam in cams:
#         pool.apply_async(run_per_folder, args=(frame_dir, barcode, cam, save_dir))
#         # run_per_folder(frame_dir, barcode, cam, save_dir)




MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(4)

# run_per_folder(frame_dir, barcode, cam, save_dir)

frame_dir = "/home/walter/big_daddy/onboard_jpg/"
save_dir = "/home/walter/big_daddy/onboard_crops/"
os.makedirs(save_dir, exist_ok=True)

barcodes_file = "/home/walter/git/pysot/barcode_p4_1.txt"
with open(barcodes_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        barcode = line.strip()
        cams = os.listdir(os.path.join(frame_dir, barcode))
        for cam in cams:
            run_per_folder(frame_dir, barcode, cam, save_dir)


