from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

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
    server_base = "http://127.0.0.1:5500"
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
    xmin = xywh_bbox[0]
    ymin = xywh_bbox[1]
    w = xywh_bbox[2]
    h = xywh_bbox[3]
    xmax = xmin + w
    ymax = ymin + h
    return [xmin, ymin, xmax, ymax]

# write yolo label label, xywh normalized form 
def save_labels(save_dir, img_name, cls_id, normalized_bbox, confidence):
    """
    save xyxy bbox to xywh yolo format
    """
    os.makedirs(save_dir, exist_ok=True)
    label_name = re.sub(".jp.+", ".txt", img_name)
    savePath = os.path.join(save_dir, label_name)
    normalized_bbox = xyxy_to_xywh(normalized_bbox)
    
    with open(savePath, 'w') as f:
        str_to_save = f"{cls_id} {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]} {confidence}"
        f.write(str_to_save)


def normalized_bbox(bbox, img_sz):
    return list(map(lambda num: num / img_sz, bbox))


def scale_bbox(n_bbox, img_sz):
    bbox = list(map(lambda num: num * img_sz, n_bbox))
    return list(map(int, bbox))



# for each folder, each camera
def run_per_folder(frames_dir, save_folder_name, read_label_from_files):
    frames = glob.glob(f"{frames_dir}/*.jp*")
    frames = sorted(frames)

    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 2
    cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)

    tracker_init = False 

    for frame in frames:
        # print(frame)
        img = cv2.imread(frame)
        bbox = []
        cls_id = 0
        confidence = 0
        if read_label_from_files:
            cls_ids, bboxs, confidences = read_img_yolo_label(frame)
        else:
            results = query_yolov5(frame)
            cls_ids, bboxs, confidences = parse_result_json(results)
        
        if len(bboxs) == 1:
            bbox = scale_bbox(bboxs[0], 324)
            xywh = xyxy_to_xywh(bbox)
            tracker = start_tracker()
            tracker.init(img, xywh)
            tracker_init = True
            color = green
            cls_id = cls_ids[0]
            confidence = confidences[0]
        elif tracker_init:
            outputs = tracker.track(img)
            bbox = outputs['bbox']
            confidence = outputs['best_score']
            bbox = xywh_to_xyxy(bbox)
            color = red
        
        
        
        if bbox:
            bbox = list(map(int, bbox))

            # draw bbox and confidence 
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            position = (100, 280)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            cv2.putText(img, str(confidence), position, font, font_scale, color, thickness)


            # save label 
            n_bbox = normalized_bbox(bbox, 324)
            save_dir = re.sub("/images", f"/{save_folder_name}", frames_dir)
            save_labels(save_dir, os.path.basename(frame), cls_id, n_bbox, confidence)

        cv2.imshow("video", img)
        cv2.waitKey(100)

frames_dir = "/home/walter/git/pysot/data/images/076150982312/cam_7"
run_per_folder(frames_dir, "test", read_label_from_files=False)

# base_dir = "/home/walter/git/pysot/data/images"
# barcodes = os.listdir(base_dir)
# for barcode in barcodes:
#     barcode_dir = os.path.join(base_dir, barcode)
#     cams = os.listdir(barcode_dir)
#     for cam in cams:
#         cam_dir = os.path.join(barcode_dir, cam)
#         run_per_folder(cam_dir, 'simple', True)